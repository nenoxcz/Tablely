import contextlib
import http.client
import io
import json
import os
import sys
import time
from pathlib import Path

import pytest

from tablely.board_view import render_status
from tablely.config import ConfigError
from tablely.gateway import Gateway, parse_gateway_config
from tablely.ledger import Ledger, process_identity
from tablely.resources import Inventory, detect_cpus

REPO = Path(__file__).resolve().parent.parent
SERVER = str(REPO / "examples" / "fake_model_server.py")
CPUS = tuple(detect_cpus())


def model(name, load=0.2, cpu=False, **kw):
    entry = {"name": name, "command": [sys.executable, SERVER, "--name", name, "--port", "{port}", "--load", str(load)]}
    if cpu:
        entry["cpu_command"] = [sys.executable, SERVER, "--name", f"{name}-cpu", "--port", "{port}", "--load", str(load)]
    entry.update(kw)
    return entry


@contextlib.contextmanager
def running_gateway(tmp_path, models, gpus=("0",), ledger=None, inventory=None, **settings):
    config = parse_gateway_config({"models": models, "listen": "127.0.0.1:0", **settings}, base_dir=tmp_path)
    inventory = inventory or Inventory(cpus=CPUS, gpus=gpus)
    gateway = Gateway(config, inventory=inventory, ledger=ledger or Ledger(tmp_path / "home"),
                      agent="gw", out=io.StringIO(), poll_interval=0.05, stop_timeout=2)
    gateway.start()
    try:
        yield gateway
    finally:
        gateway.stop()


def call(gateway, method, path, body=None, headers=None):
    host, port = gateway.httpd.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=30)
    data = json.dumps(body).encode() if body is not None else None
    connection.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    try:
        return response.status, json.loads(raw)
    except ValueError:
        return response.status, raw.decode()


def chat(gateway, name, text="hi", **extra):
    status, body = call(gateway, "POST", "/v1/chat/completions",
                        {"model": name, "messages": [{"role": "user", "content": text}], **extra})
    assert status == 200, body
    return body["choices"][0]["message"]["content"]


def pid_of(answer):
    return int(answer.split(" pid ")[1].split()[0])


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as f:
        return f.read().rsplit(")", 1)[1].split()[0] != "Z"


def wait_for(condition, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def events(gateway, kind):
    return [e for e in gateway.runner.ledger.history() if e["event"] == kind]


def test_a_request_loads_its_model_and_later_ones_reuse_it(tmp_path):
    with running_gateway(tmp_path, [model("alpha", aliases=["a"]), model("beta")], gpus=("0", "1")) as gw:
        first = chat(gw, "alpha", "hello")
        assert first.startswith("[alpha pid ") and "on GPU 0" in first and first.endswith("you said: hello")
        assert pid_of(chat(gw, "a")) == pid_of(first)  # an alias, the same server

        status, listing = call(gw, "GET", "/v1/models")
        states = {m["id"]: m["tablely"]["state"] for m in listing["data"]}
        assert states == {"alpha": "ready", "a": "ready", "beta": "idle"}
        wait_for(lambda: events(gw, "ready"))  # written to the history on the next tick
        assert events(gw, "ready")[0]["detail"].startswith("loaded in")

        view = render_status(gw.runner.ledger.snapshot(), events=gw.runner.ledger.history())
        assert "gw · local AI models" in view and "1 running · 1 idle" in view


def test_streamed_answers_are_relayed_as_they_come(tmp_path):
    with running_gateway(tmp_path, [model("alpha")]) as gw:
        host, port = gw.httpd.server_address[:2]
        connection = http.client.HTTPConnection(host, port, timeout=30)
        body = {"model": "alpha", "stream": True, "messages": [{"role": "user", "content": "one two three"}]}
        connection.request("POST", "/v1/chat/completions", body=json.dumps(body),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200 and response.getheader("Content-Type") == "text/event-stream"
        events_ = [line for line in response.read().decode().splitlines() if line.startswith("data: ")]
        assert len(events_) > 5 and events_[-1] == "data: [DONE]"
        words = "".join(json.loads(e[6:])["choices"][0]["delta"]["content"] for e in events_[:-1])
        assert words.strip().endswith("you said: one two three")


def test_an_idle_model_is_unloaded_and_comes_back_when_asked(tmp_path):
    with running_gateway(tmp_path, [model("alpha", idle="0.5s")]) as gw:
        pid = pid_of(chat(gw, "alpha"))
        wait_for(lambda: gw.status_of(gw.models["alpha"])["state"] == "idle")
        wait_for(lambda: not alive(pid) and events(gw, "unload"))
        assert "idle for" in events(gw, "unload")[0]["detail"]
        assert not gw.runner.ledger.snapshot()["jobs"]  # its GPU is free for anyone
        assert pid_of(chat(gw, "alpha")) != pid


def test_models_take_turns_on_one_gpu(tmp_path):
    with running_gateway(tmp_path, [model("alpha"), model("beta")]) as gw:
        alpha = pid_of(chat(gw, "alpha"))
        beta = chat(gw, "beta")
        assert "on GPU 0" in beta
        assert not alive(alpha)
        wait_for(lambda: events(gw, "unload"))
        assert "making room for beta" in events(gw, "unload")[0]["detail"]
        assert "on GPU 0" in chat(gw, "alpha")  # and back
        assert not alive(pid_of(beta))


def test_small_models_share_one_gpu(tmp_path):
    inventory = Inventory(cpus=CPUS, gpus=("0",), gpu_memory=(8 * 1024 ** 3,))
    with running_gateway(tmp_path, [model("alpha", gpu_share=0.5), model("beta", gpu_memory="4GiB")],
                         inventory=inventory) as gw:
        alpha, beta = chat(gw, "alpha"), chat(gw, "beta")
        assert "GPU 0 (share 0.5)" in alpha and "GPU 0 (share 0.5)" in beta
        assert alive(pid_of(alpha)) and alive(pid_of(beta)) and not events(gw, "unload")


def other_agent(ledger, *, state, priority, pid=None):
    with ledger.locked() as board:
        board.add_run("r0", {"agent": "trainer", "task": "big training", "pid": os.getpid(),
                             "pid_identity": process_identity(os.getpid()), "started_at": time.time()})
        board.put_job("r0.big", {
            "run": "r0", "agent": "trainer", "name": "big", "task": "big training", "priority": priority,
            "device": "gpu", "gpus": 1, "cpus": 1, "max_cpus": None, "command": "x", "state": state,
            "seq": board.next_seq(), "submitted_at": time.time(), "wait_reason": None, "restarts": 0,
            "allocation": {"device": "gpu", "gpus": ["0"], "cpus": [CPUS[0]]} if state == "running" else None,
            "pid": pid, "pid_identity": process_identity(pid) if pid else None, "started_at": time.time(),
        })


def test_training_takes_the_gpu_back_and_the_model_answers_from_cpu(tmp_path):
    ledger = Ledger(tmp_path / "home")
    with running_gateway(tmp_path, [model("alpha", cpu=True, priority=2)], ledger=ledger) as gw:
        on_gpu = chat(gw, "alpha")
        assert "[alpha pid" in on_gpu and "on GPU 0" in on_gpu

        other_agent(ledger, state="pending", priority=9)  # more important, wants GPU 0
        wait_for(lambda: events(gw, "unload"))
        assert not alive(pid_of(on_gpu))
        assert "making room for big (trainer)" in events(gw, "unload")[0]["detail"]

        on_cpu = chat(gw, "alpha")  # the GPU is held for "big": the model answers from the CPU
        assert on_cpu.startswith("[alpha-cpu pid") and "on cpu" in on_cpu


def test_a_gpu_only_model_waits_for_its_gpu_then_gives_up(tmp_path):
    ledger = Ledger(tmp_path / "home")
    other_agent(ledger, state="running", priority=5, pid=os.getpid())  # holds GPU 0
    with running_gateway(tmp_path, [model("alpha")], ledger=ledger, load_timeout="1s") as gw:
        status, body = call(gw, "POST", "/v1/chat/completions", {"model": "alpha", "messages": []})
        assert status == 503
        assert "not ready after 1s (waiting: needs 1 GPU(s), 0 free)" in body["error"]["message"]


def test_bad_requests_and_broken_models(tmp_path):
    broken = model("broken")
    broken["command"].append("--fail")
    with running_gateway(tmp_path, [model("alpha"), broken]) as gw:
        status, body = call(gw, "POST", "/v1/chat/completions", {"model": "gpt-9", "messages": []})
        assert status == 404 and body["error"]["code"] == "model_not_found" and "alpha" in body["error"]["message"]
        assert call(gw, "POST", "/v1/chat/completions", {"messages": []})[0] == 400
        status, body = call(gw, "POST", "/v1/chat/completions", {"model": "broken", "messages": []})
        assert status == 503
        assert "stopped before it was ready" in body["error"]["message"]
        assert "could not load model (simulated)" in body["error"]["message"]  # the end of its log
        assert call(gw, "GET", "/models/alpha/health")[0] == 200  # a model server's own API, by path


def test_api_key(tmp_path):
    with running_gateway(tmp_path, [model("alpha")], api_key="s3cret") as gw:
        assert call(gw, "GET", "/health")[0] == 200
        assert call(gw, "GET", "/v1/models")[0] == 401
        assert call(gw, "GET", "/v1/models", headers={"Authorization": "Bearer s3cret"})[0] == 200
        status, _ = call(gw, "POST", "/v1/chat/completions", {"model": "alpha", "messages": []},
                         headers={"Authorization": "Bearer wrong"})
        assert status == 401


def test_gateway_config(tmp_path):
    config = parse_gateway_config({
        "listen": "0.0.0.0:9000", "idle": "10m",
        "models": [{"name": "m", "command": "srv --port {port}", "cpu_command": ["srv", "--port", "{port}", "--cpu"],
                    "port": 7001, "aliases": ["x"], "health": "/v1/models", "warm": "~/m.gguf"}],
    }, base_dir=tmp_path)
    m = config.models[0]
    assert (config.host, config.port, config.idle) == ("0.0.0.0", 9000, 600.0)
    assert m.spec.command == "srv --port 7001" and m.spec.cpu_command == ("srv", "--port", "7001", "--cpu")
    assert m.spec.device.value == "any" and m.health == "/v1/models" and m.warm[0].endswith("/m.gguf")
    for bad, message in (
        ({"models": []}, "at least one model"),
        ({"models": [{"name": "m", "command": "x", "after": ["y"]}]}, "cannot be used for a model"),
        ({"models": [{"name": "m", "command": "x"}, {"name": "n", "command": "x", "aliases": ["m"]}]}, "used twice"),
        ({"models": [{"name": "m", "command": "x", "health": "health"}]}, "health"),
    ):
        with pytest.raises(ConfigError, match=message):
            parse_gateway_config(bad)


def test_example_gateway_file_loads():
    from tablely.gateway import load_gateway_config

    config = load_gateway_config(REPO / "examples" / "models.toml")
    assert [m.name for m in config.models] == ["qwen-coder", "embedder"]
    coder = config.models[0]
    assert coder.aliases == ("coder",) and coder.spec.device.value == "any" and str(coder.port) in coder.spec.command
