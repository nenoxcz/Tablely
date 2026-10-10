"""`tablely resume`: pick earlier work, summarize it, start the AI CLI with it."""

import json
import sys

import pytest

from tablely import cli
from tablely.cli import main
from tablely.ledger import Ledger

from test_agent_log import hook, make_repo
from test_progress import run_batch



def fake_ai(path):
    return f"{sys.executable} -c \"import json,sys; json.dump(sys.argv[1:], open({str(path)!r}, 'w'))\""


@pytest.fixture
def history(tmp_path, isolated_tablely_home, monkeypatch):
    run_batch(tmp_path, Ledger(isolated_tablely_home))  # claude-a: lr sweep, 2 ok + 1 failed
    repo = make_repo(tmp_path / "repo")
    hook(repo, "UserPromptSubmit", "c0ffee00-01", prompt="speed up the planner")
    hook(repo, "PostToolUse", "c0ffee00-01", tool_name="TaskCreate",
         tool_input={"subject": "profile share_cpus", "description": "x"}, tool_response={"task": {"id": "1"}})
    monkeypatch.chdir(repo)
    monkeypatch.delenv("TABLELY_AI", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)  # no AI CLI installed unless a test says so
    return repo


def answers(monkeypatch, *replies):
    queue = list(replies)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": queue.pop(0))


def test_picker_lists_runs_and_sessions_and_summarizes_the_pick(history, monkeypatch, capsys):
    answers(monkeypatch, "2")
    assert main(["resume"]) == 0
    out = capsys.readouterr().out
    assert "Resume which work?" in out
    assert "1  run" in out and "claude-a · lr sweep" in out and "67%" in out
    assert "2  session  c0ffee00" in out and "0/1 tasks" in out
    assert "# Resume: speed up the planner (session c0ffee00" in out
    assert "- [ ] profile share_cpus" in out
    assert "No AI CLI found" in out and "saved to" in out


def test_quit_from_the_picker(history, monkeypatch, capsys):
    answers(monkeypatch, "9", "q")
    assert main(["resume"]) == 0
    out = capsys.readouterr().out
    assert "not a number from the list" in out and "# Resume" not in out


def test_confirmed_resume_starts_the_ai_with_the_summary(history, monkeypatch, tmp_path):
    got = tmp_path / "ai.json"
    monkeypatch.setenv("TABLELY_AI", fake_ai(got))
    answers(monkeypatch, "1", "")  # pick the run, Enter = yes
    assert main(["resume"]) == 0
    (prompt,) = json.loads(got.read_text())
    assert prompt.startswith("# Resume: lr sweep (agent claude-a)") and "- [failed] c" in prompt


def test_declining_does_not_start_the_ai(history, monkeypatch, tmp_path):
    got = tmp_path / "ai.json"
    answers(monkeypatch, "n")
    assert main(["resume", "claude-a", "--ai", fake_ai(got)]) == 0
    assert not got.exists()


def test_installed_claude_is_the_default_ai(history, monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    assert main(["resume", "claude-a"]) == 0
    assert "tablely resume claude-a --ai claude --yes" in capsys.readouterr().out


def test_targets_by_agent_run_or_session(history, capsys):
    assert main(["resume", "claude-a", "--print"]) == 0
    assert capsys.readouterr().out.startswith("# Resume: lr sweep (agent claude-a)")

    run_id = next(e["run"] for e in Ledger().history() if e["event"] == "run-start")
    assert main(["resume", run_id[:4], "--print"]) == 0
    assert f"## Run {run_id}" in capsys.readouterr().out

    assert main(["resume", "c0ff", "--print"]) == 0
    assert "speed up the planner" in capsys.readouterr().out

    assert main(["resume", "nobody-xyz"]) == 2
    assert "no agent, run or coding session matches" in capsys.readouterr().err


def test_without_a_terminal_the_choices_are_listed(history, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    assert main(["resume"]) == 2
    captured = capsys.readouterr()
    assert "claude-a · lr sweep" in captured.out and "name one to resume" in captured.err


def test_yes_starts_the_ai_without_asking(history, monkeypatch, tmp_path):
    got = tmp_path / "ai.json"
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    assert main(["resume", "claude-a", "--ai", fake_ai(got), "--yes"]) == 0
    assert json.loads(got.read_text())[0].startswith("# Resume: lr sweep")
