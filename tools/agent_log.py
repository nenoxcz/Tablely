#!/usr/bin/env python3
"""Automatic work log for coding agents working on this repository.

Claude Code hooks (see .claude/settings.json) run ``agent_log.py hook`` when a
session starts, on every prompt, after every file edit, when the agent stops
and when the session ends. Each session keeps one small JSON entry:

* live, in ``.git/agent-sessions/`` (the git common dir, so every worktree of
  the clone sees it) — updated on every event, never committed;
* committed, in ``.agents/sessions/`` — one file per session, so parallel
  agents never write the same file. It is only rewritten when the agent edits
  files (or runs ``task``), i.e. when there is work to commit anyway, so the
  log never leaves the tree dirty on its own. Pushed with the work, it tells
  agents on other machines what this session was doing.

This replaces hand-written handoff notes: a new session is told, before its
first prompt, what the latest sessions were doing, where they left off and who
is busy right now.

    python3 tools/agent_log.py brief              # the handoff a new session gets
    python3 tools/agent_log.py board              # who works on what, across all branches
    python3 tools/agent_log.py board --fetch      # ... after fetching the remote first
    python3 tools/agent_log.py task "add daemon"  # state this session's task in your own words
    python3 tools/agent_log.py handoff "done X; next Y; careful with Z"   # before finishing

The task is whatever the agent last declared with ``task``; until it does, the
first line of the latest prompt stands in. When the agent stops, its last reply
is kept as an automatic handoff; ``handoff`` records one in its own words and,
unlike the automatic one, lands in the committed copy right away. The
repository may be public, so set ``AGENT_LOG_PROMPTS=0`` (e.g. under "env" in
.claude/settings.json) to keep prompts and replies out of the log entirely.

Only the standard library is used, so hooks work before anything is installed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SESSIONS_DIR = ".agents/sessions"
SHARED_DIR = "agent-sessions"  # inside the git common dir: shared by all worktrees of a clone
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
MAX_PROMPTS = 10
MAX_FILES = 40
PROMPT_CHARS = 120
REPLY_CHARS = 600
BRIEF_SESSIONS = 3  # how many recent sessions a new session is briefed on
STALE_AFTER = 6 * 3600  # an open session silent for this long is shown as stale
HIDE_ENDED_AFTER = 3 * 24 * 3600


# -- helpers -------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def age_seconds(stamp: Optional[str]) -> float:
    if not stamp:
        return float("inf")
    try:
        then = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return float("inf")
    return max(time.time() - then.timestamp(), 0)


def ago(stamp: Optional[str]) -> str:
    seconds = age_seconds(stamp)
    if seconds == float("inf"):
        return "?"
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 90 * 60:
        return f"{int(seconds // 60)}m ago"
    if seconds < 36 * 3600:
        return f"{seconds / 3600:.1f}h ago"
    return f"{int(seconds // 86400)}d ago"


def clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def first_line(text: str) -> str:
    for line in str(text).splitlines():
        if line.strip():
            return clip(line, PROMPT_CHARS)
    return ""


def prompts_enabled() -> bool:
    return os.environ.get("AGENT_LOG_PROMPTS", "1").strip().lower() not in ("0", "false", "no", "off")


def git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def repo_root(start: str) -> Path:
    top = git(Path(start), "rev-parse", "--show-toplevel")
    return Path(top) if top else Path(start)


def shared_dir(root: Path) -> Optional[Path]:
    common = git(root, "rev-parse", "--git-common-dir")
    return (root / common / SHARED_DIR) if common else None


def session_file(root: Path, session: str) -> Path:
    return root / SESSIONS_DIR / f"{session[:8]}.json"


def load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_entry(root: Path, session: str) -> Optional[Dict[str, Any]]:
    """The newest of the committed and the live copy of a session's entry."""
    copies = [load(session_file(root, session))]
    mirror = shared_dir(root)
    if mirror is not None:
        copies.append(load(mirror / f"{session[:8]}.json"))
    copies = [c for c in copies if c]
    return max(copies, key=lambda c: c.get("updated_at") or "") if copies else None


def save(root: Path, entry: Dict[str, Any], commit_copy: bool) -> None:
    """Write the live copy, and the committed copy too when ``commit_copy`` (or outside git)."""
    text = json.dumps(entry, indent=2, ensure_ascii=False) + "\n"
    targets = []
    mirror = shared_dir(root)
    if mirror is not None:
        targets.append(mirror / f"{entry['session'][:8]}.json")
    if commit_copy or mirror is None:
        targets.append(session_file(root, entry["session"]))
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)


def new_entry(session: str) -> Dict[str, Any]:
    stamp = now_iso()
    return {
        "session": session,
        "agent": os.environ.get("AGENT_LOG_NAME", "claude"),
        "branch": None,
        "status": "idle",
        "task": None,
        "task_source": None,
        "started_at": stamp,
        "updated_at": stamp,
        "start_commit": None,
        "prompts": [],
        "files": [],
        "last_reply": None,  # automatic handoff: the agent's last message when it stopped
        "handoff": None,  # explicit handoff written with `handoff`
    }


def last_reply(payload: Dict[str, Any]) -> Optional[str]:
    """The agent's final message for this turn, from the hook payload or the transcript."""
    text = payload.get("last_assistant_message")
    if isinstance(text, str) and text.strip():
        return text.strip()
    path = payload.get("transcript_path")
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(f.tell() - 512_000, 0))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("type") != "assistant":
            continue
        content = (record.get("message") or {}).get("content")
        if isinstance(content, str):
            parts = [content]
        elif isinstance(content, list):
            parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        else:
            parts = []
        text = "\n".join(p for p in parts if p.strip()).strip()
        if text:
            return text
    return None


def relative(root: Path, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    rel_text = rel.as_posix()
    return None if rel_text.startswith(".agents/") else rel_text


# -- hook ----------------------------------------------------------------------


def handle_hook(payload: Dict[str, Any]) -> None:
    event = payload.get("hook_event_name")
    session = payload.get("session_id") or os.environ.get("CLAUDE_CODE_SESSION_ID")
    if not session or not event:
        return
    root = repo_root(os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd())
    entry = load_entry(root, session) or new_entry(session)

    if event == "SessionStart":
        entry["status"] = "idle"
        entry["start_commit"] = entry.get("start_commit") or git(root, "rev-parse", "--short", "HEAD") or None
    elif event == "UserPromptSubmit":
        entry["status"] = "working"
        text = first_line(payload.get("prompt", "")) if prompts_enabled() else ""
        if text:
            entry["prompts"] = (entry.get("prompts") or [])[-(MAX_PROMPTS - 1):] + [{"at": now_iso(), "text": text}]
            if entry.get("task_source") != "agent":
                entry["task"], entry["task_source"] = text, "prompt"
    elif event == "PostToolUse":
        if payload.get("tool_name") not in EDIT_TOOLS:
            return
        tool_input = payload.get("tool_input") or {}
        rel = relative(root, tool_input.get("file_path") or tool_input.get("notebook_path"))
        if rel is None:
            return
        files = [f for f in entry.get("files") or [] if f != rel] + [rel]
        entry["files"] = files[-MAX_FILES:]
    elif event == "Stop":
        entry["status"] = "idle"
        reply = last_reply(payload) if prompts_enabled() else None
        if reply:
            entry["last_reply"] = {"at": now_iso(), "text": clip(reply, REPLY_CHARS)}
    elif event == "SessionEnd":
        entry["status"] = "ended"
    else:
        return

    entry["branch"] = git(root, "rev-parse", "--abbrev-ref", "HEAD") or entry.get("branch")
    entry["updated_at"] = now_iso()
    # Only an edit touches the committed copy: the tree is being changed anyway.
    save(root, entry, commit_copy=event == "PostToolUse")

    if event == "SessionStart":
        # SessionStart output becomes the new session's context: this is the handoff.
        text = brief(collect(root), me=session)
        if text:
            print(text)
        print('Record your task: python3 tools/agent_log.py task "<one line>". '
              'Before you finish, leave a handoff: python3 tools/agent_log.py handoff "<done / next / caveats>"')


# -- board ---------------------------------------------------------------------


def collect(root: Path) -> List[Dict[str, Any]]:
    """Sessions from this checkout, other worktrees of this clone, and every remote branch."""
    sessions: Dict[str, Dict[str, Any]] = {}

    def add(entry: Optional[Dict[str, Any]], where: str) -> None:
        if not entry or not entry.get("session"):
            return
        current = sessions.get(entry["session"])
        if current is None or (entry.get("updated_at") or "") > (current.get("updated_at") or ""):
            sessions[entry["session"]] = dict(entry, seen_in=where)

    for path in sorted((root / SESSIONS_DIR).glob("*.json")):
        add(load(path), "here")
    mirror = shared_dir(root)
    if mirror is not None:
        for path in sorted(mirror.glob("*.json")):
            add(load(path), "this machine")
    for ref in git(root, "for-each-ref", "--format=%(refname)", "refs/remotes").split():
        if ref.endswith("/HEAD"):
            continue
        short = ref[len("refs/remotes/"):]
        for name in git(root, "ls-tree", "--name-only", ref, f"{SESSIONS_DIR}/").split():
            if name.endswith(".json"):
                try:
                    add(json.loads(git(root, "show", f"{ref}:{name}")), short)
                except ValueError:
                    continue
    return sorted(sessions.values(), key=lambda s: s.get("updated_at") or "", reverse=True)


def shown_status(entry: Dict[str, Any]) -> str:
    status = entry.get("status") or "?"
    if status != "ended" and age_seconds(entry.get("updated_at")) > STALE_AFTER:
        return "stale"
    return status


def active(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in sessions if shown_status(s) in ("working", "idle")]


def brief(sessions: List[Dict[str, Any]], me: Optional[str] = None, limit: int = BRIEF_SESSIONS) -> str:
    """Where the latest sessions left off, and who is busy right now."""
    others = [s for s in sessions if s["session"] != me]
    lines: List[str] = []
    recent = [s for s in others if s.get("handoff") or s.get("last_reply") or s.get("task")][:limit]
    if recent:
        lines.append("Handoff from recent agent sessions on this repository (newest first; "
                     "working/idle ones are active now, avoid editing their files):")
    for s in recent:
        lines.append(
            f"- {s['session'][:8]} on {s.get('branch') or '?'} ({shown_status(s)}, {ago(s.get('updated_at'))}): "
            f"{clip(s.get('task') or '-', 100)}"
        )
        handoff, reply = s.get("handoff") or {}, s.get("last_reply") or {}
        if handoff:
            lines.append(f"  handoff ({ago(handoff.get('at'))}, at {handoff.get('head') or '?'}): "
                         f"{clip(handoff.get('text', ''), 500)}")
        if reply and (reply.get("at") or "") > (handoff.get("at") or ""):
            lines.append(f"  last reply ({ago(reply.get('at'))}): {clip(reply.get('text', ''), 300)}")
        files = list(reversed(s.get("files") or []))
        if files:
            lines.append(f"  files: {', '.join(files[:6])}" + (f" +{len(files) - 6}" if len(files) > 6 else ""))
    shown = {s["session"] for s in recent}
    busy = [s for s in active(others) if s["session"] not in shown]
    if busy:
        lines.append("Also active right now (avoid editing the files they are changing):")
        lines.append(render(busy))
    return "\n".join(lines)


def render(sessions: List[Dict[str, Any]]) -> str:
    headers = ["SESSION", "BRANCH", "STATUS", "UPDATED", "TASK", "FILES"]
    rows = []
    for s in sessions:
        files = s.get("files") or []
        recent = list(reversed(files))[:3]
        files_text = ", ".join(recent) + (f" +{len(files) - 3}" if len(files) > 3 else "")
        rows.append(
            [
                s["session"][:8],
                clip(s.get("branch") or "-", 32),
                shown_status(s),
                ago(s.get("updated_at")),
                clip(s.get("task") or "-", 50),
                clip(files_text or "-", 60),
            ]
        )
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    lines = ["  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    for row in rows:
        lines.append("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip())
    return "\n".join(lines)


# -- commands ------------------------------------------------------------------


def cmd_hook(args: argparse.Namespace) -> int:
    try:
        payload = json.load(sys.stdin)
        if isinstance(payload, dict):
            handle_hook(payload)
    except Exception as exc:  # a broken log must never get in the agent's way
        print(f"agent_log: {exc}", file=sys.stderr)
    return 0


def cmd_task(args: argparse.Namespace) -> int:
    root = repo_root(os.getcwd())
    session = current_session(root, args.session)
    entry = load_entry(root, session) or new_entry(session)
    entry["task"], entry["task_source"] = " ".join(args.text), "agent"
    if entry.get("status") != "ended":
        entry["status"] = "working"
    entry["branch"] = git(root, "rev-parse", "--abbrev-ref", "HEAD") or entry.get("branch")
    entry["updated_at"] = now_iso()
    save(root, entry, commit_copy=True)
    print(f"{session[:8]}: {entry['task']}")
    return 0


def current_session(root: Path, explicit: Optional[str]) -> str:
    session = explicit or os.environ.get("CLAUDE_CODE_SESSION_ID")
    if session:
        return session
    mine = [s for s in collect(root) if s.get("seen_in") == "here" and s.get("status") != "ended"]
    return mine[0]["session"] if mine else str(uuid.uuid4())


def cmd_handoff(args: argparse.Namespace) -> int:
    root = repo_root(os.getcwd())
    session = current_session(root, args.session)
    entry = load_entry(root, session) or new_entry(session)
    entry["handoff"] = {
        "at": now_iso(),
        "text": " ".join(args.text),
        "head": git(root, "rev-parse", "--short", "HEAD") or None,
    }
    entry["branch"] = git(root, "rev-parse", "--abbrev-ref", "HEAD") or entry.get("branch")
    entry["updated_at"] = now_iso()
    save(root, entry, commit_copy=True)
    print(f"{session[:8]} handoff: {entry['handoff']['text']}")
    print(f"commit {session_file(root, session).relative_to(root)} with your work so the next session gets it")
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    root = repo_root(os.getcwd())
    if args.fetch:
        git(root, "fetch", "--quiet", "--prune", "origin")
    text = brief(collect(root), me=None if args.include_me else os.environ.get("CLAUDE_CODE_SESSION_ID"),
                 limit=args.limit)
    print(text or "no agent sessions recorded yet")
    return 0


def cmd_board(args: argparse.Namespace) -> int:
    root = repo_root(os.getcwd())
    if args.fetch:
        git(root, "fetch", "--quiet", "--prune", "origin")
    sessions = collect(root)
    if not args.all:
        sessions = [
            s for s in sessions
            if shown_status(s) != "ended" or age_seconds(s.get("updated_at")) < HIDE_ENDED_AFTER
        ]
    if args.json:
        print(json.dumps(sessions, indent=2, ensure_ascii=False))
    elif sessions:
        print(render(sessions))
    else:
        print("no agent sessions recorded yet")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Work log for coding agents on this repository.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("hook", help="handle a Claude Code hook event (JSON on stdin)")
    p.set_defaults(func=cmd_hook)
    p = sub.add_parser("task", help="set this session's task in your own words")
    p.add_argument("text", nargs="+")
    p.add_argument("--session", help="session id (default: $CLAUDE_CODE_SESSION_ID)")
    p.set_defaults(func=cmd_task)
    p = sub.add_parser("handoff", help="record where this session leaves off (done / next / caveats)")
    p.add_argument("text", nargs="+")
    p.add_argument("--session", help="session id (default: $CLAUDE_CODE_SESSION_ID)")
    p.set_defaults(func=cmd_handoff)
    p = sub.add_parser("brief", help="the handoff a new session gets: recent sessions and who is busy")
    p.add_argument("--fetch", action="store_true", help="git fetch first to see other machines' latest pushes")
    p.add_argument("--limit", type=int, default=BRIEF_SESSIONS, help="how many recent sessions to show")
    p.add_argument("--include-me", action="store_true", help="also show the current session")
    p.set_defaults(func=cmd_brief)
    p = sub.add_parser("board", help="show who is working on what")
    p.add_argument("--all", action="store_true", help="include sessions that ended long ago")
    p.add_argument("--fetch", action="store_true", help="git fetch first to see other machines' latest pushes")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_board)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
