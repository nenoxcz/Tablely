import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "agent_log.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
}


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env={**os.environ, **GIT_ENV})


def make_repo(path):
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    git(path, "commit", "-q", "--allow-empty", "-m", "init")
    return path


def hook(repo, event, session, **fields):
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(repo)}
    env.pop("AGENT_LOG_PROMPTS", None)
    env.update(fields.pop("env", {}))
    payload = {"hook_event_name": event, "session_id": session, **fields}
    result = subprocess.run([sys.executable, str(TOOL), "hook"], input=json.dumps(payload),
                            capture_output=True, text=True, env=env, cwd=repo)
    assert result.returncode == 0, result.stderr
    return result.stdout


def tool(repo, *args, env=None):
    result = subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True,
                            cwd=repo, env={**os.environ, **(env or {})})
    assert result.returncode == 0, result.stderr
    return result.stdout


def committed(repo, session):
    path = repo / ".agents" / "sessions" / f"{session[:8]}.json"
    return json.loads(path.read_text()) if path.exists() else None


def entry(repo, session):
    """The newest copy: the live one under .git/ or the committed one."""
    live = json.loads((repo / ".git" / "agent-sessions" / f"{session[:8]}.json").read_text())
    tree = committed(repo, session)
    return max([live] + ([tree] if tree else []), key=lambda e: e["updated_at"])


def dirty(repo):
    return subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout


def test_hooks_record_a_session_lifecycle(tmp_path):
    repo = make_repo(tmp_path / "repo")
    sid = "aaaaaaaa-0001"
    hook(repo, "SessionStart", sid)
    assert entry(repo, sid)["status"] == "idle"
    hook(repo, "UserPromptSubmit", sid, prompt="Add a daemon\nmore detail")
    hook(repo, "PostToolUse", sid, tool_name="Edit", tool_input={"file_path": str(repo / "a.py")})
    hook(repo, "PostToolUse", sid, tool_name="Write", tool_input={"file_path": str(repo / "b.py")})
    hook(repo, "PostToolUse", sid, tool_name="Edit", tool_input={"file_path": str(repo / "a.py")})
    hook(repo, "PostToolUse", sid, tool_name="Edit", tool_input={"file_path": "/elsewhere/x.py"})
    hook(repo, "PostToolUse", sid, tool_name="Bash", tool_input={"command": "ls"})
    e = entry(repo, sid)
    assert e["status"] == "working" and e["branch"] == "main"
    assert e["task"] == "Add a daemon" and e["task_source"] == "prompt"
    assert e["files"] == ["b.py", "a.py"]  # most recent last, no duplicates, nothing outside the repo
    hook(repo, "Stop", sid)
    assert entry(repo, sid)["status"] == "idle"
    hook(repo, "SessionEnd", sid)
    assert entry(repo, sid)["status"] == "ended"


def test_only_edits_touch_the_committed_copy(tmp_path):
    # Status and prompt updates must never leave the tree dirty on their own,
    # or every finished turn would ask for a commit.
    repo = make_repo(tmp_path / "repo")
    sid = "abababab-0007"
    hook(repo, "SessionStart", sid)
    hook(repo, "UserPromptSubmit", sid, prompt="what does the planner do?")
    hook(repo, "Stop", sid)
    assert dirty(repo) == ""
    assert entry(repo, sid)["task"] == "what does the planner do?"

    hook(repo, "UserPromptSubmit", sid, prompt="fix the planner tie-break")
    hook(repo, "PostToolUse", sid, tool_name="Edit", tool_input={"file_path": str(repo / "planner.py")})
    saved = committed(repo, sid)
    assert saved["task"] == "fix the planner tie-break" and saved["files"] == ["planner.py"]
    git(repo, "add", ".agents")
    git(repo, "commit", "-q", "-m", "work")
    hook(repo, "Stop", sid)
    hook(repo, "SessionEnd", sid)
    assert dirty(repo) == ""
    assert entry(repo, sid)["status"] == "ended"


def test_declared_task_wins_over_later_prompts(tmp_path):
    repo = make_repo(tmp_path / "repo")
    sid = "bbbbbbbb-0002"
    hook(repo, "UserPromptSubmit", sid, prompt="first idea")
    tool(repo, "task", "NUMA-aware", "placement", env={"CLAUDE_CODE_SESSION_ID": sid})
    hook(repo, "UserPromptSubmit", sid, prompt="ok continue")
    e = entry(repo, sid)
    assert e["task"] == "NUMA-aware placement" and e["task_source"] == "agent"
    assert [p["text"] for p in e["prompts"]] == ["first idea", "ok continue"]


def test_prompts_can_be_kept_out_of_the_log(tmp_path):
    repo = make_repo(tmp_path / "repo")
    hook(repo, "UserPromptSubmit", "cccccccc-3", prompt="secret plan", env={"AGENT_LOG_PROMPTS": "0"})
    hook(repo, "PostToolUse", "cccccccc-3", tool_name="Edit", tool_input={"file_path": str(repo / "a.py")})
    assert "secret" not in json.dumps(entry(repo, "cccccccc-3"))
    assert "secret" not in json.dumps(committed(repo, "cccccccc-3"))


def test_bad_input_never_blocks_the_agent(tmp_path):
    repo = make_repo(tmp_path / "repo")
    result = subprocess.run([sys.executable, str(TOOL), "hook"], input="not json", capture_output=True,
                            text=True, cwd=repo)
    assert result.returncode == 0


def test_sessions_on_other_clones_and_worktrees_are_visible(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    first = make_repo(tmp_path / "first")
    git(first, "remote", "add", "origin", str(remote))
    git(first, "checkout", "-q", "-b", "feat/daemon")
    hook(first, "UserPromptSubmit", "dddddddd-0004", prompt="build the submit daemon")
    hook(first, "PostToolUse", "dddddddd-0004", tool_name="Edit",
         tool_input={"file_path": str(first / "tablely" / "daemon.py")})
    git(first, "add", ".agents")
    git(first, "commit", "-q", "-m", "wip")
    git(first, "push", "-q", "origin", "feat/daemon")

    # another machine: a fresh clone sees the pushed session on start
    second = tmp_path / "second"
    subprocess.run(["git", "clone", "-q", str(remote), str(second)], check=True, capture_output=True)
    out = hook(second, "SessionStart", "eeeeeeee-0005")
    assert "dddddddd" in out and "build the submit daemon" in out and "tablely/daemon.py" in out
    assert "eeeeeeee" not in out  # a session is not briefed about itself

    # same machine, another worktree: seen live, before anything is committed
    worktree = tmp_path / "wt"
    git(first, "worktree", "add", "-q", str(worktree), "-b", "feat/other")
    hook(worktree, "UserPromptSubmit", "ffffffff-0006", prompt="tune affinity")
    board = tool(first, "board")
    assert "ffffffff" in board and "tune affinity" in board and "feat/other" in board


def test_last_reply_is_kept_as_an_automatic_handoff(tmp_path):
    repo = make_repo(tmp_path / "repo")
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join(json.dumps(r) for r in [
        {"type": "user", "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Started."}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "name": "Bash"}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Done: planner tie-break fixed. Next: NUMA placement."}]}},
    ]) + "\n")
    hook(repo, "Stop", "a1a1a1a1-0008", transcript_path=str(transcript))
    assert entry(repo, "a1a1a1a1-0008")["last_reply"]["text"] == "Done: planner tie-break fixed. Next: NUMA placement."
    assert dirty(repo) == ""  # kept live only; the tree stays clean

    hook(repo, "Stop", "b2b2b2b2-0009", transcript_path=str(transcript), env={"AGENT_LOG_PROMPTS": "0"})
    assert entry(repo, "b2b2b2b2-0009")["last_reply"] is None

    out = hook(repo, "SessionStart", "c3c3c3c3-0010")
    assert "last reply" in out and "Next: NUMA placement." in out


def test_explicit_handoff_reaches_the_next_session_on_another_machine(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    first = make_repo(tmp_path / "first")
    git(first, "remote", "add", "origin", str(remote))
    sid = "d4d4d4d4-0011"
    hook(first, "SessionStart", sid)
    tool(first, "task", "speed up the planner", env={"CLAUDE_CODE_SESSION_ID": sid})
    out = tool(first, "handoff", "done: share_cpus is O(n log n); next: benchmark 256 cores",
               env={"CLAUDE_CODE_SESSION_ID": sid})
    assert "commit .agents/sessions/d4d4d4d4.json" in out
    assert committed(first, sid)["handoff"]["text"].startswith("done: share_cpus")
    git(first, "add", ".agents")
    git(first, "commit", "-q", "-m", "work")
    git(first, "push", "-q", "origin", "main")

    second = tmp_path / "second"
    subprocess.run(["git", "clone", "-q", str(remote), str(second)], check=True, capture_output=True)
    out = hook(second, "SessionStart", "e5e5e5e5-0012")
    assert "speed up the planner" in out
    assert "handoff (" in out and "next: benchmark 256 cores" in out
    briefing = tool(second, "brief", env={"CLAUDE_CODE_SESSION_ID": "e5e5e5e5-0012"})
    assert briefing.count("d4d4d4d4") == 1 and "e5e5e5e5" not in briefing
