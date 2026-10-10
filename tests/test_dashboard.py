import json
import sys
import time
import urllib.error
import urllib.request

import pytest

from tablely.dashboard import Dashboard, serve_in_thread
from tablely.ledger import Ledger

from test_agent_log import hook, make_repo
from test_progress import run_batch


def call(url, body=None, headers=None):
    request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode()


@pytest.fixture
def ui(tmp_path, isolated_tablely_home):
    ledger = Ledger(isolated_tablely_home)
    run_batch(tmp_path, ledger)  # claude-a: two ok, one failed
    repo = make_repo(tmp_path / "repo")
    hook(repo, "UserPromptSubmit", "s1s1s1s1-01", prompt="add a dashboard")
    hook(repo, "PostToolUse", "s1s1s1s1-01", tool_name="TaskCreate",
         tool_input={"subject": "server", "description": "x"}, tool_response={"task": {"id": "1"}})
    hook(repo, "PostToolUse", "s1s1s1s1-01", tool_name="TaskUpdate", tool_input={"taskId": "1", "status": "completed"})
    dashboard = Dashboard(ledger, repo=str(repo))
    server, base = serve_in_thread(dashboard)
    yield dashboard, base
    server.shutdown()
    server.server_close()


def test_page_carries_the_token_and_state_has_rates(ui):
    dashboard, base = ui
    status, page = call(base + "/")
    assert status == 200 and dashboard.token in page and "재개하기" in page
    status, body = call(base + "/api/state")
    state = json.loads(body)
    (run,) = state["runs"]
    assert run["agent"] == "claude-a" and abs(run["achievement"] - 2 / 3) < 1e-9
    assert run["counts"]["ok"] == 2 and run["counts"]["failed"] == 1
    assert state["agents"][0]["name"] == "claude-a"
    (session,) = state["sessions"]
    assert (session["done"], session["total"], session["task"]) == (1, 1, "add a dashboard")


def test_resume_requires_the_token_and_a_local_host(ui):
    dashboard, base = ui
    assert call(base + "/api/resume", {"kind": "agent", "id": "claude-a"})[0] == 403
    assert call(base + "/api/resume", {"kind": "agent", "id": "claude-a"}, {"X-Tablely-Token": "nope"})[0] == 403
    assert call(base + "/api/state", headers={"Host": "attacker.example"})[0] == 403
    token = {"X-Tablely-Token": dashboard.token}
    assert call(base + "/api/resume", {"kind": "planet", "id": "x"}, token)[0] == 400
    assert call(base + "/api/resume", {"kind": "agent", "id": "nobody"}, token)[0] == 404


def test_resume_buttons_return_summaries(ui):
    dashboard, base = ui
    token = {"X-Tablely-Token": dashboard.token}
    status, body = call(base + "/api/resume", {"kind": "agent", "id": "claude-a"}, token)
    result = json.loads(body)
    assert status == 200 and not result["launched"]
    assert result["prompt"].startswith("# Resume: lr sweep (agent claude-a)")
    assert open(result["prompt_file"]).read() == result["prompt"]

    run_id = json.loads(call(base + "/api/state")[1])["runs"][0]["run"]
    result = json.loads(call(base + "/api/resume", {"kind": "run", "id": run_id}, token)[1])
    assert f"## Run {run_id}" in result["prompt"]

    result = json.loads(call(base + "/api/resume", {"kind": "session", "id": "s1s1"}, token)[1])
    assert "- [x] server" in result["prompt"] and "add a dashboard" in result["prompt"]


def test_resume_can_start_the_configured_ai(ui, tmp_path):
    dashboard, base = ui
    got = tmp_path / "got.json"
    dashboard.resume_command = (
        f"{sys.executable} -c \"import json,sys; json.dump(sys.argv[1:], open({str(got)!r}, 'w'))\""
    )
    result = json.loads(call(base + "/api/resume", {"kind": "agent", "id": "claude-a"},
                             {"X-Tablely-Token": dashboard.token})[1])
    assert result["launched"] and result["pid"]
    for _ in range(250):
        if got.exists():
            break
        time.sleep(0.02)
    assert json.loads(got.read_text())[0].startswith("# Resume: lr sweep")
