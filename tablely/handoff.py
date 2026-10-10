"""What the CLI and the MCP server share: the status view, and finding work to
resume and summarizing it for an AI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import _fmt, worklog
from .board_view import render_status
from .ledger import Ledger
from .progress import RunView, collect_runs, resume_prompt, session_prompt


@dataclass
class Choice:
    kind: str  # "run", "agent" or "session"
    ident: str
    line: str  # how it is listed for picking


def status_text(ledger: Ledger, repo: Optional[str], hours: float = 12.0, color: bool = False) -> str:
    sessions = worklog.recent_sessions(repo, hours) if repo else []
    return render_status(ledger.snapshot(), events=ledger.history(), sessions=sessions, hours=hours, color=color)


def resumable(ledger: Ledger, repo: Optional[str], hours: float) -> Tuple[List[RunView], List[Dict[str, Any]]]:
    """Runs (live first, then those that ended within ``hours``) and coding sessions."""
    since = time.time() - hours * 3600
    runs = [r for r in collect_runs(ledger.snapshot(), ledger.history()) if r.live or (r.ended or 0) >= since]
    return runs, (worklog.recent_sessions(repo, hours) if repo else [])


def choices(runs: Sequence[RunView], sessions: Sequence[Dict[str, Any]]) -> List[Choice]:
    return [_run_choice(r) for r in runs] + [_session_choice(s) for s in sessions]


def match(target: str, runs: Sequence[RunView], repo: Optional[str]) -> Choice:
    """An agent name, a run id (prefix) or a coding-session id (prefix). Raises LookupError."""
    if any(r.agent == target for r in runs):
        return Choice("agent", target, target)
    matches = sorted({r.run for r in runs if r.run.startswith(target)})
    if len(matches) > 1:
        raise LookupError(f"{target!r} matches several runs: {', '.join(matches)}")
    if matches:
        return Choice("run", matches[0], matches[0])
    session = worklog.find_session(repo, target) if repo else None
    if session is not None:
        return Choice("session", session["session"], session["session"])
    raise LookupError(f"no agent, run or coding session matches {target!r}")


def summarize(
    ledger: Ledger, choice: Choice, runs: Sequence[RunView], repo: Optional[str], hours: float
) -> Tuple[str, str, Optional[str]]:
    """``(summary, label for its file, directory the AI should start in)``."""
    if choice.kind == "session":
        session = worklog.find_session(repo, choice.ident)
        return session_prompt(session), f"session-{choice.ident[:8]}", str(worklog.repo_root(repo))
    picked = [r for r in runs if (r.run if choice.kind == "run" else r.agent) == choice.ident]
    if choice.kind == "run" and not picked:  # an older run, outside the window
        picked = [r for r in collect_runs(ledger.snapshot(), ledger.history()) if r.run == choice.ident]
    if not picked:
        raise LookupError(f"no {choice.kind} {choice.ident!r}")
    since = time.time() - hours * 3600
    notes = [e for e in ledger.history(agent=picked[0].agent)
             if e.get("event") == "note" and e.get("t", 0) >= since][-10:][::-1]
    return resume_prompt(picked, notes, agent=picked[0].agent), f"{choice.kind}-{choice.ident}", None


def _run_choice(run: RunView) -> Choice:
    state = "running" if run.live else f"ended {_fmt.duration(time.time() - run.ended)} ago"
    head = _fmt.pad(_fmt.clip(f"{run.agent} · {run.task or '-'}", 44), 44)
    return Choice("run", run.run, f"run      {run.run}  {head}  {_fmt.bar(run.achievement, 12)} "
                                  f"{_fmt.percent(run.achievement)}  {state}")


def _session_choice(session: Dict[str, Any]) -> Choice:
    todos = session.get("todos") or []
    done = sum(1 for t in todos if t.get("status") == "completed")
    fraction = done / len(todos) if todos else None
    head = _fmt.pad(_fmt.clip(f"coding · {session.get('task') or '-'}", 44), 44)
    tasks = f"{done}/{len(todos)} tasks · " if todos else ""
    return Choice("session", session["session"],
                  f"session  {session['session'][:8]}  {head}  {_fmt.bar(fraction, 12)} "
                  f"{_fmt.percent(fraction)}  {tasks}{session.get('status', '?')}")
