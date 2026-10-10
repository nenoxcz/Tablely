"""tablely mcp: the MCP server AI apps use to refer to Tablely."""

import json
import os
import stat
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

from tablely.ledger import Ledger
from tablely.mcp import PROTOCOL_VERSIONS, McpServer, load_token, make_http_server

from test_agent_log import hook, make_repo
from test_progress import run_batch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def server(tmp_path, isolated_tablely_home):
    run_batch(tmp_path, Ledger(isolated_tablely_home))  # claude-a: lr sweep, 2 ok + 1 failed
    repo = make_repo(tmp_path / "repo")
    hook(repo, "UserPromptSubmit", "deadbeef-01", prompt="add an MCP server")
    return McpServer(Ledger(isolated_tablely_home), repo=str(repo))


def rpc(server, method, params=None, ident=1):
    message = {"jsonrpc": "2.0", "id": ident, "method": method}
    if params is not None:
        message["params"] = params
    return server.handle(message)


def call(server, tool, **arguments):
    result = rpc(server, "tools/call", {"name": tool, "arguments": arguments})["result"]
    return result["content"][0]["text"], result["isError"]


def test_initialize_negotiates_the_protocol_version(server):
    result = rpc(server, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                        "clientInfo": {"name": "claude-ai", "version": "1"}})["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert set(result["capabilities"]) == {"tools", "resources", "prompts"}
    assert result["serverInfo"]["name"] == "tablely" and "tablely_resume" in result["instructions"]
    newer = rpc(server, "initialize", {"protocolVersion": "2099-01-01", "capabilities": {}})["result"]
    assert newer["protocolVersion"] == PROTOCOL_VERSIONS[0]


def test_unknown_methods_notifications_and_bad_messages(server):
    assert rpc(server, "server/discover", {})["error"]["code"] == -32601  # newer clients then fall back
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert server.handle({"id": 1, "method": "ping"})["error"]["code"] == -32600
    assert rpc(server, "ping") == {"jsonrpc": "2.0", "id": 1, "result": {}}
    batch = server.handle_raw([{"jsonrpc": "2.0", "id": 7, "method": "ping"},
                               {"jsonrpc": "2.0", "method": "notifications/cancelled"}])
    assert batch == [{"jsonrpc": "2.0", "id": 7, "result": {}}]


def test_tools_are_described_with_schemas(server):
    tools = {t["name"]: t for t in rpc(server, "tools/list")["result"]["tools"]}
    assert set(tools) == {"tablely_status", "tablely_resume", "tablely_brief", "tablely_history", "tablely_note"}
    for tool in tools.values():
        assert tool["inputSchema"]["type"] == "object" and tool["description"]
    assert tools["tablely_note"]["inputSchema"]["required"] == ["text"]
    assert tools["tablely_status"]["annotations"]["readOnlyHint"] is True
    assert tools["tablely_note"]["annotations"]["readOnlyHint"] is False


def test_status_resume_brief_and_history_tools(server):
    status, error = call(server, "tablely_status", hours=24)
    assert not error and "claude-a · lr sweep" in status and "add an MCP server" in status

    choices, _ = call(server, "tablely_resume")
    assert "claude-a" in choices and "deadbeef" in choices

    summary, error = call(server, "tablely_resume", target="claude-a")
    assert not error and summary.startswith("# Resume: lr sweep (agent claude-a)")
    session, _ = call(server, "tablely_resume", target="deadbeef")
    assert "add an MCP server" in session

    missing, error = call(server, "tablely_resume", target="nobody-xyz")
    assert error and "no agent, run or coding session" in missing
    bad, error = call(server, "tablely_status", hours=-1)
    assert error and "hours" in bad

    brief, _ = call(server, "tablely_brief")
    assert "## Progress" in brief and "lr sweep" in brief
    history, _ = call(server, "tablely_history", agent="claude-a", limit=3)
    assert "claude-a" in history and "run-end" in history


def test_note_tool_records_for_other_agents(server):
    rpc(server, "initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                               "clientInfo": {"name": "chatgpt", "version": "1"}})
    text, error = call(server, "tablely_note", text="  try lr 3e-4 next  ")
    assert not error and text == "noted for chatgpt: try lr 3e-4 next"
    status, _ = call(server, "tablely_status")
    assert "chatgpt" in status and "try lr 3e-4 next" in status
    _, error = call(server, "tablely_note", text="  ")
    assert error
    assert rpc(server, "tools/call", {"name": "tablely_launch", "arguments": {}})["error"]["code"] == -32602


def test_resources_and_resume_prompt(server):
    uris = [r["uri"] for r in rpc(server, "resources/list")["result"]["resources"]]
    assert uris == ["tablely://status", "tablely://brief"]
    content = rpc(server, "resources/read", {"uri": "tablely://brief"})["result"]["contents"][0]
    assert content["mimeType"] == "text/markdown" and content["text"].startswith("# Tablely brief")
    assert rpc(server, "resources/read", {"uri": "tablely://nope"})["error"]["code"] == -32002

    prompts = rpc(server, "prompts/list")["result"]["prompts"]
    assert prompts[0]["name"] == "resume"
    got = rpc(server, "prompts/get", {"name": "resume", "arguments": {"target": "claude-a"}})["result"]
    assert got["messages"][0]["role"] == "user"
    assert got["messages"][0]["content"]["text"].startswith("# Resume: lr sweep")


def test_stdio_transport_speaks_only_json(tmp_path, isolated_tablely_home):
    run_batch(tmp_path, Ledger(isolated_tablely_home))
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "tablely_resume",
                                                                       "arguments": {"target": "claude-a"}}},
    ]
    stdin = "\n".join(json.dumps(m) for m in lines) + "\nnot json\n"
    result = subprocess.run([sys.executable, "-m", "tablely", "mcp"], input=stdin, capture_output=True,
                            text=True, cwd=REPO_ROOT, env=dict(os.environ, TABLELY_HOME=str(isolated_tablely_home)),
                            timeout=30)
    replies = [json.loads(line) for line in result.stdout.splitlines()]  # every stdout line is JSON
    assert [r.get("id") for r in replies] == [1, 2, None]
    assert replies[0]["result"]["protocolVersion"] == "2025-03-26"
    assert replies[1]["result"]["content"][0]["text"].startswith("# Resume: lr sweep")
    assert replies[2]["error"]["code"] == -32700


@pytest.fixture
def http(server):
    httpd = make_http_server(server, "127.0.0.1", 0, "s3cret")
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def post(url, body, headers=None, method="POST"):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json",
                                              "Accept": "application/json, text/event-stream", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode(), dict(response.headers)
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode(), dict(err.headers)


def test_http_transport_requires_the_token(http):
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    assert post(f"{http}/mcp/wrong", ping)[0] == 404
    assert post(f"{http}/mcp", ping)[0] == 404
    assert post(f"{http}/mcp", ping, {"Authorization": "Bearer s3cret"})[0] == 200
    status, body, headers = post(f"{http}/mcp/s3cret", {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "claude-ai"}}})
    assert status == 200 and json.loads(body)["result"]["protocolVersion"] == "2025-11-25"
    assert headers.get("Mcp-Session-Id")
    assert post(f"{http}/mcp/s3cret", {"jsonrpc": "2.0", "method": "notifications/initialized"})[0] == 202
    assert post(f"{http}/mcp/s3cret", None, method="GET")[0] == 405
    status, body, _ = post(f"{http}/mcp/s3cret", {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                                  "params": {"name": "tablely_status", "arguments": {}}})
    assert status == 200 and "claude-a" in json.loads(body)["result"]["content"][0]["text"]


def test_token_is_kept_private_and_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("TABLELY_MCP_TOKEN", raising=False)
    first = load_token(tmp_path)
    assert first == load_token(tmp_path) and len(first) >= 20
    assert stat.S_IMODE(os.stat(tmp_path / "mcp-token").st_mode) == 0o600
    assert load_token(tmp_path, "explicit") == "explicit"
    monkeypatch.setenv("TABLELY_MCP_TOKEN", "from-env")
    assert load_token(tmp_path) == "from-env"
