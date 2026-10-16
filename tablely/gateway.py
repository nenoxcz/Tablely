"""Local AI gateway: one OpenAI-compatible endpoint for local models, swapped in and out of GPUs.

    tablely serve models.toml                 # http://127.0.0.1:8800/v1

Each model is a server process (llama.cpp's ``llama-server``, vLLM, ...) that
Tablely runs as one of its jobs, so models and training share the machine by
the same rules:

* **swap in on demand**: a request for a model that is not loaded queues its
  server; it starts as soon as the planner gives it a GPU (or CPU cores,
  with ``cpu_command``), and the request is forwarded once the server
  answers its health check;
* **swap out when idle**: a model unused for ``idle`` stops and frees its GPU;
* **take turns**: when a requested model waits for a GPU that an idle model
  of this gateway holds, the least recently used idle model is unloaded;
* **yield to training**: when a more important job (any agent) waits for a
  GPU an idle model holds, the model is unloaded; while that job runs, a
  model with ``cpu_command`` answers from the CPU instead;
* **warm RAM**: model files listed in ``warm`` are read ahead into the page
  cache, so loading again comes from RAM rather than disk.

Requests choose the model with the ``"model"`` field of OpenAI-style JSON
bodies (``/v1/chat/completions``, ``/v1/completions``, ``/v1/embeddings`` ...);
``/models/<name>/<path>`` reaches any other API of a model server as is.
Responses, including streamed ones, are relayed as they arrive.
"""

from __future__ import annotations

import glob
import hmac
import http.client
import json
import os
import re
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, TextIO, Tuple, Union

from . import _fmt
from .config import Config, ConfigError, _check_keys, _mapping, parse_config, read_file
from .ledger import Board, Ledger, MemoryLedger
from .planner import Plan
from .resources import Inventory
from .runner import JobState, Runner
from .spec import JobSpec, parse_duration

DEFAULT_LISTEN = ("127.0.0.1", 8800)
_TOP_KEYS = {"listen", "idle", "load_timeout", "api_key", "task", "log_dir", "backfill", "resources", "models"}
_MODEL_ONLY_KEYS = {"aliases", "health", "port", "upstream_name", "idle", "warm"}
_NOT_FOR_MODELS = {"after", "matrix", "timeout", "switchable", "max_gpus"}
_HOP_BY_HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade",
               "host", "content-length"}


class GatewayError(Exception):
    def __init__(self, status: int, message: str, code: str = "model_unavailable") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(eq=False)
class Model:
    """One model server and what the gateway knows about it right now."""

    spec: JobSpec
    port: int
    aliases: Tuple[str, ...] = ()
    health: str = "/health"
    upstream_name: Optional[str] = None  # rewrite the request's "model" field to this
    idle: Optional[float] = None  # seconds unused before it is unloaded (None: the gateway default)
    warm: Tuple[str, ...] = ()  # files to keep in the page cache
    inflight: int = 0  # requests being served or waiting for this model
    last_used: float = 0.0
    ready_pid: Optional[int] = None  # the server process that passed its health check

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass
class GatewayConfig:
    models: List[Model]
    jobs: Config  # resources, policy and log dir, parsed like a job file
    host: str = DEFAULT_LISTEN[0]
    port: int = DEFAULT_LISTEN[1]
    idle: float = 600.0
    load_timeout: float = 300.0
    api_key: Optional[str] = None


# -- configuration -------------------------------------------------------------------


def load_gateway_config(path: Union[str, Path]) -> GatewayConfig:
    path = Path(path)
    return parse_gateway_config(read_file(path), base_dir=path.parent)


def parse_gateway_config(data: Any, base_dir: Union[str, Path] = ".") -> GatewayConfig:
    data = _mapping(data, "gateway file")
    _check_keys(data, _TOP_KEYS, "gateway file")
    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ConfigError("gateway file must define at least one model under 'models'")

    jobs, extras = [], []
    taken_ports = set()
    for i, raw in enumerate(raw_models, start=1):
        raw = dict(_mapping(raw, f"model #{i}"))
        where = f"model {raw.get('name', f'#{i}')!r}"
        bad = sorted(set(raw) & _NOT_FOR_MODELS)
        if bad:
            raise ConfigError(f"{where}: {', '.join(bad)} cannot be used for a model")
        extra = {key: raw.pop(key) for key in list(raw) if key in _MODEL_ONLY_KEYS}
        port = extra.get("port")
        if port is None:
            port = _free_port(taken_ports)
        elif isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
            raise ConfigError(f"{where}: port must be a TCP port number")
        taken_ports.add(port)
        for key in ("command", "cpu_command"):
            if key in raw:
                raw[key] = _fill_port(raw[key], port)
        if "cpu_command" in raw:
            raw.setdefault("device", "any")
        jobs.append(raw)
        extras.append((where, port, extra))

    task = data.get("task", "local AI models")
    config = parse_config(
        {"jobs": jobs, "resources": data.get("resources", {}), "backfill": data.get("backfill", False),
         "log_dir": data.get("log_dir", "tablely-logs"), "task": task},
        base_dir=base_dir,
    )
    models = [_model(spec, where, port, extra) for spec, (where, port, extra) in zip(config.jobs, extras)]
    names: Dict[str, str] = {}
    for model in models:
        for alias in (model.name, *model.aliases):
            if alias in names:
                raise ConfigError(f"model name {alias!r} is used twice ({names[alias]} and {model.name})")
            names[alias] = model.name

    host, port = DEFAULT_LISTEN
    if "listen" in data:
        host, port = parse_listen(data["listen"])
    api_key = data.get("api_key")
    if api_key is not None and (not isinstance(api_key, str) or not api_key):
        raise ConfigError("api_key must be a non-empty string")
    return GatewayConfig(
        models=models, jobs=config, host=host, port=port,
        idle=_seconds(data.get("idle", 600), "idle"),
        load_timeout=_seconds(data.get("load_timeout", 300), "load_timeout"),
        api_key=api_key,
    )


def _model(spec: JobSpec, where: str, port: int, extra: Mapping[str, Any]) -> Model:
    aliases = extra.get("aliases", [])
    if not isinstance(aliases, list) or not all(isinstance(a, str) and a for a in aliases):
        raise ConfigError(f"{where}: aliases must be a list of names")
    health = extra.get("health", "/health")
    if not isinstance(health, str) or not health.startswith("/"):
        raise ConfigError(f'{where}: health must be a path such as "/health"')
    upstream = extra.get("upstream_name")
    if upstream is not None and not isinstance(upstream, str):
        raise ConfigError(f"{where}: upstream_name must be a string")
    warm = extra.get("warm", [])
    if isinstance(warm, str):
        warm = [warm]
    if not isinstance(warm, list) or not all(isinstance(w, str) for w in warm):
        raise ConfigError(f"{where}: warm must be a list of file paths")
    idle = _seconds(extra["idle"], f"{where}: idle") if "idle" in extra else None
    return Model(spec=spec, port=port, aliases=tuple(aliases), health=health, upstream_name=upstream,
                 idle=idle, warm=tuple(os.path.expanduser(w) for w in warm))


def parse_listen(text: Any) -> Tuple[str, int]:
    """``"127.0.0.1:8800"``, ``":8800"`` or ``8800``."""
    text = str(text)
    host, _, port = text.rpartition(":")
    try:
        number = int(port)
    except ValueError:
        raise ConfigError(f"listen must be HOST:PORT, got {text!r}") from None
    return host.strip("[]") or DEFAULT_LISTEN[0], number


def _seconds(value: Any, what: str) -> float:
    try:
        return parse_duration(value)
    except ValueError as exc:
        raise ConfigError(f"{what}: {exc}") from None


def _fill_port(command: Any, port: int) -> Any:
    def fill(text: Any) -> Any:
        return re.sub(r"\{port\}", str(port), text) if isinstance(text, str) else text

    return [fill(part) for part in command] if isinstance(command, list) else fill(command)


def _free_port(taken: set) -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port not in taken:
            return port


def warm_files(patterns: Sequence[str]) -> int:
    """Ask the kernel to read files into the page cache (in the background). Returns bytes asked for."""
    total = 0
    for pattern in patterns:
        for path in glob.glob(pattern) or [pattern]:
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                size = os.fstat(fd).st_size
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
                total += size
            finally:
                os.close(fd)
    return total


# -- the gateway ---------------------------------------------------------------------


class Gateway:
    """Runs the model servers through Tablely and answers HTTP requests for them."""

    def __init__(
        self,
        config: GatewayConfig,
        *,
        inventory: Optional[Inventory] = None,
        ledger: Union[Ledger, MemoryLedger, None] = None,
        agent: Optional[str] = None,
        out: Optional[TextIO] = None,
        poll_interval: float = 0.25,
        stop_timeout: float = 10.0,
        base_env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.config = config
        self.models: Dict[str, Model] = {m.name: m for m in config.models}
        self._names = {alias: m for m in config.models for alias in (m.name, *m.aliases)}
        jobs = config.jobs
        self.runner = Runner(
            inventory if inventory is not None else jobs.inventory(),
            [m.spec for m in config.models],
            policy=jobs.policy, log_dir=jobs.log_dir, poll_interval=poll_interval, stop_timeout=stop_timeout,
            out=out, base_env=base_env, ledger=ledger, agent=agent, task=jobs.task, mps=jobs.mps,
            on_demand=True,
        )
        self.runner.after_tick = self._make_room
        self._lock = threading.Lock()  # Model.inflight / last_used / ready_pid
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self.httpd: Optional[ThreadingHTTPServer] = None

    # -- lifecycle --------------------------------------------------------------

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> str:
        """Start serving in background threads; returns the base URL."""
        for model in self.models.values():
            warm_files(model.warm)
        self.runner._register()  # raises here if the machine's shared pool cannot fit these models
        self.httpd = ThreadingHTTPServer((self.config.host, self.config.port), _handler(self))
        self.httpd.daemon_threads = True
        for target, name in ((self._run_jobs, "tablely-jobs"), (self._monitor, "tablely-models"),
                             (lambda: self.httpd.serve_forever(poll_interval=0.2), "tablely-http")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        return self.url

    def stop(self) -> None:
        """Stop answering, stop every model server and leave the ledger."""
        self._stop.set()
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        for thread in self._threads:
            thread.join(timeout=self.runner.stop_timeout + 5)

    def _run_jobs(self) -> None:
        try:
            self.runner.serve(self._stop, registered=True)
        except Exception as exc:  # say so instead of letting requests wait for nothing
            self.runner._say(f"gateway stopped: scheduling failed: {exc!r}")
            self._stop.set()

    # -- models -----------------------------------------------------------------

    def resolve(self, name: Any) -> Model:
        model = self._names.get(name) if isinstance(name, str) else None
        if model is None:
            known = ", ".join(sorted(self._names))
            raise GatewayError(404, f"no model named {name!r} here; available: {known}", "model_not_found")
        return model

    def state(self, model: Model) -> Dict[str, Any]:
        """What the model is doing: idle, waiting, loading, ready (or failed), and where."""
        with self.runner._mutex:
            rec = self.runner.records[model.name]
            pid = rec.process.pid if rec.process is not None else None
            info = {
                "state": rec.state.value, "pid": pid, "placement": _fmt.placement(rec.allocation),
                "wait_reason": rec.wait_reason, "error": rec.error, "ended_at": rec.ended_at,
                "started_at": rec.started_at, "log": str(rec.log_path) if rec.log_path else None,
            }
        if info["state"] == "running":
            info["state"] = "ready" if pid is not None and model.ready_pid == pid else "loading"
        elif info["state"] == "pending":
            info["state"] = "waiting"
        return info

    def acquire(self, model: Model) -> None:
        """Make sure ``model``'s server is up, counting this request as in flight until ``release``."""
        with self._lock:
            model.inflight += 1
            model.last_used = time.time()
        try:
            self._wait_ready(model)
        except BaseException:
            self.release(model)
            raise

    def release(self, model: Model) -> None:
        with self._lock:
            model.inflight -= 1
            model.last_used = time.time()

    def _wait_ready(self, model: Model) -> None:
        deadline = time.time() + self.config.load_timeout
        requested: Optional[float] = None
        while True:
            if self._stop.is_set():
                raise GatewayError(503, "the gateway is shutting down")
            info = self.state(model)
            state = info["state"]
            if state == "ready":
                return
            if state in ("idle", "succeeded", "failed", "cancelled"):
                ended = info["ended_at"]
                fresh = requested is not None and ended is not None and ended >= requested
                if fresh and state in ("succeeded", "failed"):
                    why = info["error"] or "its server exited"
                    raise GatewayError(503, f"model {model.name} stopped before it was ready ({why}); "
                                            f"see {info['log']}{_log_tail(info['log'])}")
                if requested is None or fresh:
                    requested = time.time()
                    self.runner.request(model.name)
            if time.time() >= deadline:
                what = f"waiting: {info['wait_reason']}" if state == "waiting" else state
                raise GatewayError(503, f"model {model.name} is not ready after "
                                        f"{_fmt.duration(self.config.load_timeout)} ({what})")
            time.sleep(0.05)

    def _monitor(self) -> None:
        """Health-check loading servers; unload models idle for too long."""
        while not self._stop.wait(min(self.runner.poll_interval, 0.25)):
            for model in self.models.values():
                info = self.state(model)
                if info["state"] == "loading" and _healthy(model):
                    with self._lock:
                        model.ready_pid = info["pid"]
                    took = _fmt.duration(time.time() - (info["started_at"] or time.time()))
                    self.runner.note(model.name, "ready", f"loaded in {took} on {info['placement']}")
                elif info["state"] == "ready":
                    idle_after = model.idle if model.idle is not None else self.config.idle
                    with self._lock:
                        idle_for = time.time() - model.last_used if model.inflight == 0 else 0.0
                    if idle_for >= idle_after:
                        self.runner.unload(model.name, f"idle for {_fmt.duration(idle_for)}")
                        warm_files(model.warm)

    def _make_room(self, board: Board, plan: Plan) -> None:
        """Unload an idle model when another job waits for what it holds (runs inside the runner's tick).

        A more important job of any agent may take a model's GPU (or RAM); a
        model of this gateway may take another model's. Least recently used
        goes first, one at a time.
        """
        records = self.runner.records
        if any(records[m.name].stop_sent_at is not None for m in self.models.values()):
            return  # an unload is under way; look again once it is done
        with self._lock:
            idle = [
                m for m in self.models.values()
                if m.inflight == 0 and records[m.name].state is JobState.RUNNING and records[m.name].allocation
            ]
        if not idle:
            return
        ours = {self.runner._keys[m.name]: m for m in self.models.values()}
        waiting = sorted(
            ((key, job) for key, job in board.jobs.items() if job["state"] == "pending" and job.get("wait_reason")),
            key=lambda kv: (-kv[1]["priority"], kv[1]["seq"]),
        )
        for key, job in waiting:
            reason = job["wait_reason"]
            needs_gpu, needs_ram = "GPU" in reason, "RAM" in reason
            if not needs_gpu and not needs_ram:
                continue
            candidates = [
                m for m in idle
                if (key in ours or m.spec.priority < job["priority"])
                and ((needs_gpu and records[m.name].allocation.on_gpu) or (needs_ram and m.spec.ram))
            ]
            if candidates:
                victim = min(candidates, key=lambda m: m.last_used)
                who = job["name"] if key in ours else f"{job['name']} ({job['agent']})"
                self.runner.unload(victim.name, f"making room for {who}")
                return

    def status_of(self, model: Model) -> Dict[str, Any]:
        info = self.state(model)
        return {
            "state": info["state"],
            "placement": info["placement"] if info["state"] in ("ready", "loading") else None,
            "wait_reason": info["wait_reason"] if info["state"] == "waiting" else None,
            "inflight": model.inflight, "last_used": model.last_used or None, "port": model.port,
            "priority": model.spec.priority, "aliases": list(model.aliases),
        }

    def status(self) -> Dict[str, Any]:
        models = [dict(name=m.name, **self.status_of(m)) for m in self.models.values()]
        return {"agent": self.runner.agent, "run": self.runner.run_id, "models": models}

    # -- requests ---------------------------------------------------------------

    def forward(self, handler: "BaseHTTPRequestHandler", model: Model, path: str, body: Optional[bytes]) -> None:
        """Send the request to the model's server and relay the answer as it streams back."""
        headers = {k: v for k, v in handler.headers.items() if k.lower() not in _HOP_BY_HOP}
        if self.config.api_key:  # the gateway's key is not the model server's business
            headers = {k: v for k, v in headers.items() if k.lower() not in ("authorization", "x-api-key")}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        upstream = http.client.HTTPConnection("127.0.0.1", model.port, timeout=3600)
        try:
            upstream.request(handler.command, path, body=body, headers=headers)
            response = upstream.getresponse()
            handler.send_response(response.status, response.reason)
            handler.relaying = True  # from here on an error can only cut the answer short
            for key, value in response.getheaders():
                if key.lower() not in _HOP_BY_HOP:
                    handler.send_header(key, value)
            length = response.getheader("Content-Length")
            chunked = length is None
            if chunked:
                handler.send_header("Transfer-Encoding", "chunked")
            else:
                handler.send_header("Content-Length", length)
            handler.end_headers()
            read = getattr(response, "read1", response.read)
            while True:
                data = read(65536)
                if not data:
                    break
                handler.wfile.write(b"%x\r\n%s\r\n" % (len(data), data) if chunked else data)
                handler.wfile.flush()
            if chunked:
                handler.wfile.write(b"0\r\n\r\n")
        finally:
            upstream.close()


def _healthy(model: Model) -> bool:
    connection = http.client.HTTPConnection("127.0.0.1", model.port, timeout=2)
    try:
        connection.request("GET", model.health)
        return connection.getresponse().status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _log_tail(path: Optional[str], lines: int = 5) -> str:
    try:
        tail = Path(path).read_text(errors="replace").splitlines()[-lines:]
    except (OSError, TypeError):
        return ""
    return (": " + " | ".join(line.strip() for line in tail if line.strip())) if tail else ""


def _handler(gateway: Gateway) -> type:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "tablely-gateway"

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def _send_json(self, status: int, data: Any) -> None:
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str, code: str) -> None:
            kind = "invalid_request_error" if status < 500 else "server_error"
            self._send_json(status, {"error": {"message": message, "type": kind, "code": code}})

        def _authorized(self) -> bool:
            key = gateway.config.api_key
            if not key:
                return True
            given = self.headers.get("x-api-key") or ""
            auth = self.headers.get("Authorization") or ""
            if auth.lower().startswith("bearer "):
                given = auth[7:].strip()
            if hmac.compare_digest(given.encode(), key.encode()):
                return True
            self._error(401, "missing or wrong API key", "invalid_api_key")
            return False

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/health":
                return self._send_json(200, {"status": "ok"})
            if not self._authorized():
                return
            if path in ("/v1/models", "/models"):
                data = [{"id": name, "object": "model", "owned_by": "tablely",
                         "tablely": {"model": m.name, **{k: v for k, v in gateway.status_of(m).items()}}}
                        for name, m in sorted(gateway._names.items())]
                return self._send_json(200, {"object": "list", "data": data})
            if path == "/status":
                return self._send_json(200, gateway.status())
            self._dispatch(None)

        def do_POST(self) -> None:
            if not self._authorized():
                return
            length = self.headers.get("Content-Length")
            if length is None:
                return self._error(411, "send the body with a Content-Length", "length_required")
            self._dispatch(self.rfile.read(int(length)))

        do_PUT = do_POST

        def do_DELETE(self) -> None:
            if self._authorized():
                self._dispatch(None)

        def _dispatch(self, body: Optional[bytes]) -> None:
            path = self.path
            try:
                direct = re.match(r"^/models/([^/]+)(/.*)?$", path)
                if direct:  # /models/<name>/<anything>: the model server's own API
                    model, path = gateway.resolve(direct.group(1)), direct.group(2) or "/"
                elif path.startswith("/v1/") and body is not None:
                    try:
                        request = json.loads(body or b"{}")
                    except ValueError:
                        raise GatewayError(400, "the body is not JSON", "invalid_json") from None
                    if not isinstance(request, dict) or "model" not in request:
                        raise GatewayError(400, 'say which model to use with "model"', "model_required")
                    model = gateway.resolve(request["model"])
                    if model.upstream_name and request["model"] != model.upstream_name:
                        request["model"] = model.upstream_name
                        body = json.dumps(request).encode()
                else:
                    raise GatewayError(404, f"nothing at {self.command} {path}", "not_found")
                gateway.acquire(model)
            except GatewayError as exc:
                return self._error(exc.status, str(exc), exc.code)
            self.relaying = False
            try:
                gateway.forward(self, model, path, body)
            except (OSError, http.client.HTTPException) as exc:
                self.close_connection = True
                if not self.relaying:  # nothing sent yet: say what went wrong
                    try:
                        self._error(502, f"model {model.name} did not answer: {exc}", "bad_gateway")
                    except OSError:
                        pass
            finally:
                gateway.release(model)

    return Handler
