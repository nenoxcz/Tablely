"""``tablely mcp``: Tablely as an MCP server, so AI apps can refer to it.

* Claude Desktop, Claude Code and other MCP clients start ``tablely mcp``
  themselves (stdio) — on this machine, or on a GPU server through
  ``ssh gpu-server tablely mcp``.
* Apps that only take remote connectors — the Claude app on the web or a
  phone, the ChatGPT app — reach ``tablely mcp --http`` through an HTTPS
  tunnel. That endpoint requires a secret token, kept in
  ``$TABLELY_HOME/mcp-token``.

Tools: ``tablely_status``, ``tablely_brief``, ``tablely_resume``,
``tablely_history`` (read-only) and ``tablely_note`` (leaves a note). Resources
``tablely://status`` and ``tablely://brief``; prompt ``resume``. Nothing here
starts or stops jobs.

The handshake-era protocol (2024-11-05 to 2025-11-25) is implemented with the
standard library; newer clients fall back to it after their discovery probe.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple

from . import __version__, handoff
from .board_view import render_brief, render_history
from .ledger import Ledger, make_event

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")  # newest first

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602
RESOURCE_NOT_FOUND = -32002

INSTRUCTIONS = (
    "Tablely shares one machine's GPUs and CPU cores between training jobs by priority, and records what "
    "every agent (AI or person) runs there. Use tablely_status to see progress (completion bars per run and "
    "job, coding sessions, notes), tablely_resume to get a summary of earlier work you can continue from, "
    "tablely_brief for a short handoff, tablely_history for past events, and tablely_note to leave a note "
    "for whoever works on this next."
)


class McpError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _schema(properties: Dict[str, Any], required: Tuple[str, ...] = ()) -> Dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


HOURS = {"type": "number", "minimum": 0, "description": "How far back to look, in hours."}

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "tablely_status",
        "title": "Tablely status",
        "description": "Progress of all work on the Tablely machine: every live run with a completion bar per "
                       "job (device, cores, latest progress), runs that finished recently, coding sessions with "
                       "their task-list completion, and the agents' notes.",
        "inputSchema": _schema({"hours": dict(HOURS, default=12)}),
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "tablely_resume",
        "title": "Resume earlier work",
        "description": "A summary of earlier work another AI can continue from: goal, completion, results and "
                       "last metrics, failures with log paths, what is still running, notes and next steps. "
                       "Without a target it lists what can be resumed (runs, agents, coding sessions).",
        "inputSchema": _schema({
            "target": {"type": "string",
                       "description": "Agent name, run id or coding-session id (a prefix is enough)."},
            "hours": dict(HOURS, default=72),
        }),
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "tablely_brief",
        "title": "Tablely handoff brief",
        "description": "Short handoff of the training side: completion per run, what runs now, what finished "
                       "(with last metrics) and the agents' notes.",
        "inputSchema": _schema({
            "hours": dict(HOURS, default=24),
            "agent": {"type": "string", "description": "Only this agent's work."},
        }),
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "tablely_history",
        "title": "Tablely history",
        "description": "Past events: job starts, finishes, failures, device switches, notes.",
        "inputSchema": _schema({
            "agent": {"type": "string", "description": "Only this agent."},
            "job": {"type": "string", "description": "Only this job name."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 30,
                      "description": "Last N events."},
        }),
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "tablely_note",
        "title": "Leave a note",
        "description": "Record what you are working on or what should happen next. Other agents see it in "
                       "status, briefs and resume summaries.",
        "inputSchema": _schema({
            "text": {"type": "string", "description": "The note."},
            "agent": {"type": "string", "description": "Who is writing (default: this app's name)."},
        }, required=("text",)),
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False,
                        "openWorldHint": False},
    },
]

RESOURCES = [
    {"uri": "tablely://status", "name": "status", "title": "Tablely status", "mimeType": "text/plain",
     "description": "Progress of all work on the machine (same as the tablely_status tool)."},
    {"uri": "tablely://brief", "name": "brief", "title": "Tablely handoff brief", "mimeType": "text/markdown",
     "description": "Handoff for the next agent (same as the tablely_brief tool)."},
]

PROMPTS = [
    {
        "name": "resume",
        "title": "Resume Tablely work",
        "description": "Start from a summary of earlier work (a run, an agent's runs, or a coding session).",
        "arguments": [{"name": "target", "required": False,
                       "description": "Agent name, run id or coding-session id; empty lists the choices."}],
    }
]


class McpServer:
    """Answers MCP JSON-RPC messages from the ledger (and a repository's work log)."""

    def __init__(self, ledger: Ledger, repo: Optional[str] = None) -> None:
        self.ledger = ledger
        self.repo = repo
        self.client_name = "mcp-client"
        self._methods: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            "initialize": self._initialize,
            "ping": lambda params: {},
            "tools/list": lambda params: {"tools": TOOLS},
            "tools/call": self._call_tool,
            "resources/list": lambda params: {"resources": RESOURCES},
            "resources/templates/list": lambda params: {"resourceTemplates": []},
            "resources/read": self._read_resource,
            "prompts/list": lambda params: {"prompts": PROMPTS},
            "prompts/get": self._get_prompt,
        }

    # -- JSON-RPC ---------------------------------------------------------------

    def handle_raw(self, data: Any) -> Any:
        """A decoded message or batch -> the response(s), or None when nothing is owed."""
        if isinstance(data, list):
            responses = [r for r in (self.handle(m) for m in data) if r is not None]
            return responses or None
        return self.handle(data)

    def handle(self, message: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "expected a JSON-RPC 2.0 message")
        if "method" not in message:
            return None  # a response to something we never send
        is_request = "id" in message
        params = message.get("params") or {}
        try:
            method = self._methods.get(message["method"])
            if method is None:
                raise McpError(METHOD_NOT_FOUND, f"method not found: {message['method']}")
            if not isinstance(params, dict):
                raise McpError(INVALID_PARAMS, "params must be an object")
            result = method(params)
        except McpError as exc:
            return _error(message.get("id"), exc.code, str(exc)) if is_request else None
        except Exception as exc:  # keep serving; report the failure to this request only
            return _error(message.get("id"), -32603, f"internal error: {exc}") if is_request else None
        return {"jsonrpc": "2.0", "id": message["id"], "result": result} if is_request else None

    # -- methods ----------------------------------------------------------------

    def _initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        requested = params.get("protocolVersion")
        info = params.get("clientInfo") or {}
        if isinstance(info, dict) and info.get("name"):
            self.client_name = str(info["name"])
        return {
            "protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
            "capabilities": {"tools": {"listChanged": False}, "resources": {"listChanged": False},
                             "prompts": {"listChanged": False}},
            "serverInfo": {"name": "tablely", "title": "Tablely", "version": __version__},
            "instructions": INSTRUCTIONS,
        }

    def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name, args = params.get("name"), params.get("arguments") or {}
        tools = {
            "tablely_status": self._status,
            "tablely_resume": self._resume,
            "tablely_brief": self._brief,
            "tablely_history": self._history,
            "tablely_note": self._note,
        }
        if name not in tools:
            raise McpError(INVALID_PARAMS, f"unknown tool: {name}")
        if not isinstance(args, dict):
            raise McpError(INVALID_PARAMS, "arguments must be an object")
        try:
            text = tools[name](args)
        except (LookupError, ValueError, TypeError) as exc:  # a problem with the call, told to the model
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        return {"content": [{"type": "text", "text": text}], "isError": False}

    def _read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        uri = params.get("uri")
        readers = {"tablely://status": (self._status, "text/plain"), "tablely://brief": (self._brief, "text/markdown")}
        if uri not in readers:
            raise McpError(RESOURCE_NOT_FOUND, f"resource not found: {uri}")
        read, mime = readers[uri]
        return {"contents": [{"uri": uri, "mimeType": mime, "text": read({})}]}

    def _get_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if params.get("name") != "resume":
            raise McpError(INVALID_PARAMS, f"unknown prompt: {params.get('name')}")
        target = (params.get("arguments") or {}).get("target") or None
        try:
            text = self._resume({"target": target} if target else {})
        except LookupError as exc:
            raise McpError(INVALID_PARAMS, str(exc)) from None
        return {"description": "Summary of earlier Tablely work to continue from",
                "messages": [{"role": "user", "content": {"type": "text", "text": text}}]}

    # -- tools ------------------------------------------------------------------

    def _status(self, args: Dict[str, Any]) -> str:
        return handoff.status_text(self.ledger, self.repo, _hours(args, 12))

    def _resume(self, args: Dict[str, Any]) -> str:
        hours = _hours(args, 72)
        runs, sessions = handoff.resumable(self.ledger, self.repo, hours)
        target = args.get("target")
        if not target:
            options = handoff.choices(runs, sessions)
            if not options:
                return f"Nothing to resume: no runs in the last {hours:g}h and no coding sessions."
            agents = sorted({r.agent for r in runs})
            return ("What can be resumed (call tablely_resume again with target = a run id, a session id, or an "
                    f"agent name: {', '.join(agents) or '-'}):\n" + "\n".join(c.line for c in options))
        choice = handoff.match(str(target), runs, self.repo)
        prompt, _, _ = handoff.summarize(self.ledger, choice, runs, self.repo, hours)
        return prompt

    def _brief(self, args: Dict[str, Any]) -> str:
        hours = _hours(args, 24)
        events = self.ledger.history(agent=args.get("agent"))
        return render_brief(self.ledger.snapshot(), events, hours=hours, agent=args.get("agent"))

    def _history(self, args: Dict[str, Any]) -> str:
        limit = args.get("limit", 30)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer from 1 to 500")
        return render_history(self.ledger.history(agent=args.get("agent"), job=args.get("job"), limit=limit))

    def _note(self, args: Dict[str, Any]) -> str:
        text = " ".join(str(args.get("text") or "").split())
        if not text:
            raise ValueError("text is empty")
        agent = str(args.get("agent") or self.client_name)
        with self.ledger.locked() as board:
            board.set_note(agent, text)
            board.log(make_event(time.time(), "note", agent, None, None, None, text))
        return f"noted for {agent}: {text}"


def _hours(args: Dict[str, Any], default: float) -> float:
    hours = args.get("hours", default)
    if isinstance(hours, bool) or not isinstance(hours, (int, float)) or hours < 0:
        raise ValueError("hours must be a non-negative number")
    return float(hours)


def _error(ident: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}


# -- transports --------------------------------------------------------------------


def serve_stdio(server: McpServer, stdin: Optional[BinaryIO] = None, stdout: Optional[BinaryIO] = None) -> None:
    """Newline-delimited JSON-RPC on stdin/stdout (nothing else may be written to stdout)."""
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer
    for raw in stdin:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            response: Any = _error(None, PARSE_ERROR, "invalid JSON")
        else:
            response = server.handle_raw(data)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
            stdout.flush()


def load_token(home: Path, explicit: Optional[str] = None) -> str:
    """The HTTP endpoint's secret: --token, $TABLELY_MCP_TOKEN, or one kept in $TABLELY_HOME/mcp-token."""
    token = explicit or os.environ.get("TABLELY_MCP_TOKEN")
    if token:
        return token
    path = home / "mcp-token"
    try:
        token = path.read_text().strip()
    except OSError:
        token = ""
    if not token:
        home.mkdir(parents=True, exist_ok=True)
        token = secrets.token_urlsafe(24)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(token + "\n")
    return token


def make_http_server(server: McpServer, host: str, port: int, token: str) -> ThreadingHTTPServer:
    """Streamable HTTP (JSON responses only) at ``/mcp/<token>``, or ``/mcp`` with ``Authorization: Bearer``."""
    lock = threading.Lock()  # one message at a time: the ledger is file-locked anyway

    class Handler(BaseHTTPRequestHandler):
        server_version = "tablely-mcp"

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def _authorized(self) -> bool:
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == f"/mcp/{token}":
                return True
            bearer = self.headers.get("Authorization", "")
            return path == "/mcp" and secrets.compare_digest(bearer, f"Bearer {token}")

        def _reply(self, status: int, body: Optional[Any] = None, headers: Optional[Dict[str, str]] = None) -> None:
            data = b"" if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            if body is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(min(length, 1 << 20)) or b"null")
            except ValueError:
                return self._reply(HTTPStatus.BAD_REQUEST, _error(None, PARSE_ERROR, "invalid JSON"))
            with lock:
                response = server.handle_raw(data)
            if response is None:
                return self._reply(HTTPStatus.ACCEPTED)
            headers = {}
            if isinstance(data, dict) and data.get("method") == "initialize":
                headers["Mcp-Session-Id"] = secrets.token_hex(16)
            self._reply(HTTPStatus.OK, response, headers)

        def do_GET(self) -> None:  # noqa: N802 - no server-to-client stream
            if not self._authorized():
                return self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
            self._reply(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "use POST"}, {"Allow": "POST"})

        def do_DELETE(self) -> None:  # noqa: N802 - sessions hold no state to end
            if not self._authorized():
                return self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
            self._reply(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "sessions are stateless"}, {"Allow": "POST"})

    return ThreadingHTTPServer((host, port), Handler)
