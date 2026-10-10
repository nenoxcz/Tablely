"""Completion rates and resume prompts, rebuilt from the board and the history.

A *run* is one ``tablely run`` (one agent, one job file). Its completion rate
counts finished-OK jobs fully and running jobs by the fraction they reported
(``client.progress``), out of all its jobs. ``resume_prompt`` turns runs (or a
coding session from the work log) into a summary another AI can continue from.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from . import _fmt
from .ledger import RUNNING, allocation_from_json

RESULT_STATES = {"done": "ok", "fail": "failed", "stop": "cancelled"}


@dataclass
class JobView:
    name: str
    state: str = "waiting"  # waiting | running | ok | failed | cancelled
    task: Optional[str] = None
    fraction: Optional[float] = None  # 0..1 as reported; 1.0 once ok
    progress: Optional[str] = None
    placement: Optional[str] = None  # "GPU 0,1" / "CPU"
    detail: Optional[str] = None  # wait reason, failure, duration
    seconds: Optional[float] = None
    git: Optional[Dict[str, Any]] = None
    log: Optional[str] = None
    switches: int = 0

    @property
    def credit(self) -> float:
        """How much this job counts toward its run's completion (0..1)."""
        if self.state == "ok":
            return 1.0
        if self.state == "running":
            return self.fraction or 0.0
        return 0.0


@dataclass
class RunView:
    run: str
    agent: str
    task: Optional[str] = None
    started: Optional[float] = None
    ended: Optional[float] = None  # None while live
    jobs: List[JobView] = field(default_factory=list)

    @property
    def live(self) -> bool:
        return self.ended is None

    @property
    def achievement(self) -> float:
        return sum(j.credit for j in self.jobs) / len(self.jobs) if self.jobs else 0.0

    def count(self, state: str) -> int:
        return sum(1 for j in self.jobs if j.state == state)

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(live=self.live, achievement=self.achievement,
                    counts={s: self.count(s) for s in ("ok", "failed", "cancelled", "running", "waiting")})
        return data


def collect_runs(data: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> List[RunView]:
    """Every run seen in the history or on the board, newest first."""
    runs: Dict[str, RunView] = {}

    def job(view: RunView, name: str) -> JobView:
        for j in view.jobs:
            if j.name == name:
                return j
        j = JobView(name=name)
        view.jobs.append(j)
        return j

    for e in events:
        rid, kind = e.get("run"), e.get("event")
        if not rid:
            continue
        view = runs.get(rid)
        if view is None:
            view = runs[rid] = RunView(run=rid, agent=e.get("agent") or "?", task=None, started=e.get("t"))
        if kind == "run-start":
            view.task, view.started = e.get("task"), e.get("t")
            names = e.get("jobs") or _names_from_detail(e.get("detail"))
            for name in names:
                job(view, name).task = (e.get("job_tasks") or {}).get(name) or view.task
        elif kind == "run-end" or kind == "run-lost":
            view.ended = e.get("t")
        elif e.get("job"):
            j = job(view, e["job"])
            j.task = e.get("task") or j.task
            if kind == "start":
                j.state, j.detail = "running", None
                j.placement = _placement(e)
            elif kind in RESULT_STATES:
                j.state = RESULT_STATES[kind]
                j.progress = e.get("progress") or j.progress
                j.fraction = 1.0 if kind == "done" else e.get("progress_frac", j.fraction)
                j.seconds, j.git, j.log = e.get("seconds"), e.get("git"), e.get("log")
                j.detail = e.get("detail")
            elif kind == "requeue":
                j.state, j.switches, j.detail = "waiting", j.switches + 1, e.get("detail")
            elif kind == "wait":
                if j.state == "waiting":
                    j.detail = e.get("detail")

    # The live board knows more about runs still going: progress, wait reasons, orphans.
    for rid, run in data.get("runs", {}).items():
        view = runs.get(rid) or runs.setdefault(rid, RunView(run=rid, agent=run.get("agent", "?"),
                                                             task=run.get("task"), started=run.get("started_at")))
        view.ended = None
        for name in run.get("jobs", []):
            job(view, name)
    for entry in data.get("jobs", {}).values():
        view = runs.get(entry["run"])
        if view is None:  # an orphan whose run is gone
            view = runs[entry["run"]] = RunView(run=entry["run"], agent=entry.get("agent", "?"),
                                                task=entry.get("task"))
        j = job(view, entry["name"])
        j.task = entry.get("task") or j.task
        j.switches = entry.get("restarts", j.switches)
        if entry["state"] == RUNNING:
            j.state = "running"
            j.placement = _fmt.placement(allocation_from_json(entry.get("allocation")))
            j.progress, j.fraction = entry.get("progress"), entry.get("progress_frac")
            j.detail = "runner gone" if entry.get("orphan") else None
        else:
            j.state, j.detail = "waiting", entry.get("wait_reason")
    for view in runs.values():
        if not view.live:
            for j in view.jobs:
                if j.state in ("waiting", "running"):  # the run ended before this job finished
                    j.state = "cancelled"
    return sorted(runs.values(), key=lambda r: (not r.live, -(r.started or 0)))


def agent_achievement(runs: Iterable[RunView]) -> float:
    jobs = [j for r in runs for j in r.jobs]
    return sum(j.credit for j in jobs) / len(jobs) if jobs else 0.0


def _names_from_detail(detail: Optional[str]) -> List[str]:
    # older run-start events only had "3 job(s): a, b, c"
    if not detail or ": " not in detail:
        return []
    return [n.strip() for n in detail.split(": ", 1)[1].split(",") if n.strip()]


def _placement(event: Mapping[str, Any]) -> str:
    if event.get("device") == "gpu":
        return _fmt.gpu_placement(event.get("gpus") or [], event.get("gpu_share"))
    return "CPU"


def _pct(fraction: Optional[float]) -> str:
    return f"{fraction:.0%}" if fraction is not None else "?"


def _job_line(j: JobView) -> str:
    label = {"running": f"running {_pct(j.fraction)}" if j.fraction is not None else "running"}.get(j.state, j.state)
    line = f"- [{label}] {j.name}"
    if j.task:
        line += f" ({j.task})"
    bits = []
    if j.placement and j.state == "running":
        bits.append(f"on {j.placement}")
    if j.state == "ok" and j.seconds is not None:
        bits.append(f"took {_fmt.duration(j.seconds)}")
    if j.progress:
        bits.append(("last progress: " if j.state != "running" else "progress: ") + j.progress)
    if j.state in ("failed", "waiting", "cancelled") and j.detail:
        bits.append(j.detail)
    if j.git:
        bits.append(f"code {j.git.get('commit')}{'+dirty' if j.git.get('dirty') else ''}"
                    + (f" on {j.git['branch']}" if j.git.get("branch") else ""))
    if j.state == "failed" and j.log and j.log not in (j.detail or ""):
        bits.append(f"log {j.log}")
    if j.switches:
        bits.append(f"moved between CPU/GPU {j.switches}x")
    return line + (": " + "; ".join(bits) if bits else "")


def resume_prompt(
    runs: Sequence[RunView],
    notes: Sequence[Mapping[str, Any]] = (),
    agent: Optional[str] = None,
    now: Optional[float] = None,
) -> str:
    """A summary of training work another AI can pick up from."""
    now = time.time() if now is None else now
    who = agent or (runs[0].agent if runs else "an agent")
    tasks = []
    for r in runs:
        if r.task and r.task not in tasks:
            tasks.append(r.task)
    jobs = [j for r in runs for j in r.jobs]
    lines = [
        f"# Resume: {'; '.join(tasks) or 'Tablely work'} (agent {who})",
        "",
        f"You are picking up machine-learning work that agent {who} ran on this machine with Tablely "
        "(a scheduler that shares the GPUs and CPU cores between jobs by priority). Here is where it stands.",
        "",
        f"Overall: {agent_achievement(runs):.0%} done. {sum(j.state == 'ok' for j in jobs)} of {len(jobs)} jobs "
        f"finished OK, {sum(j.state == 'failed' for j in jobs)} failed, "
        f"{sum(j.state == 'running' for j in jobs)} running, {sum(j.state == 'waiting' for j in jobs)} waiting.",
    ]
    for r in runs:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.started)) if r.started else "?"
        state = "still running" if r.live else f"ended {_fmt.duration(now - r.ended)} ago"
        lines += ["", f"## Run {r.run}: {r.task or '-'} ({r.achievement:.0%}; started {when}, {state})"]
        lines += [_job_line(j) for j in r.jobs] or ["- (no jobs)"]
    if notes:
        lines += ["", "## Notes the agent left (newest first)"]
        lines += [f"- {_clock(n)} {n.get('detail')}" for n in notes]
    lines += ["", "## How to continue"]
    if any(j.state == "failed" for j in jobs):
        lines.append("- Read the logs of failed jobs, fix the cause, and run them again with `tablely run <jobfile>`.")
    if any(j.state in ("running", "waiting") for j in jobs):
        lines.append("- Running and waiting jobs continue on their own; follow them with `tablely status`.")
    if notes:
        lines.append("- The notes above say what the previous agent meant to do next; start there unless it is done.")
    lines.append('- Record what you are doing with `tablely note --agent <your name> "..."` '
                 "so the next agent can resume too.")
    return "\n".join(lines)


def session_prompt(session: Mapping[str, Any], now: Optional[float] = None) -> str:
    """A summary of a coding session (from the work log) another AI can pick up from."""
    todos = session.get("todos") or []
    done = sum(1 for t in todos if t.get("status") == "completed")
    head = f"# Resume: {session.get('task') or 'coding work'} (session {session['session'][:8]}"
    head += f" on branch {session['branch']})" if session.get("branch") else ")"
    lines = [
        head,
        "",
        "You are continuing work an earlier coding session did on this repository. Here is where it stands.",
    ]
    if todos:
        lines += ["", f"## Task list ({done}/{len(todos)} done, {done / len(todos):.0%})"]
        marks = {"completed": "[x]", "in_progress": "[~]"}
        lines += [f"- {marks.get(t.get('status'), '[ ]')} {t.get('subject')}" for t in todos]
    handoff, reply = session.get("handoff") or {}, session.get("last_reply") or {}
    if handoff or reply:
        lines += ["", "## Where it left off"]
        if handoff:
            lines.append(f"- handoff note ({handoff.get('at')}, at commit {handoff.get('head') or '?'}): "
                         f"{handoff.get('text')}")
        if reply and (reply.get("at") or "") > (handoff.get("at") or ""):
            lines.append(f"- its last message ({reply.get('at')}): {reply.get('text')}")
    prompts = session.get("prompts") or []
    if prompts:
        lines += ["", "## What the user asked (oldest first)"]
        lines += [f"- {p.get('text')}" for p in prompts]
    files = list(reversed(session.get("files") or []))
    if files:
        lines += ["", "## Files it changed (most recent first)", "- " + ", ".join(files[:20])]
    lines += ["", "## How to continue"]
    if session.get("branch"):
        lines.append(f"- Work on branch `{session['branch']}`"
                     + (f"; `git log --oneline {session['start_commit']}..` shows what it committed."
                        if session.get("start_commit") else "."))
    if todos and done < len(todos):
        lines.append("- Pick up the first task not marked [x]; [~] was in progress when it stopped.")
    lines.append("- Run the tests before and after your changes, and leave your own handoff when you stop.")
    return "\n".join(lines)


def _clock(event: Mapping[str, Any]) -> str:
    stamp = str(event.get("time", ""))
    return stamp.replace("T", " ")[:16]


def save_prompt(prompt: str, directory: Path, label: str) -> Path:
    """Keep a copy of a resume summary (``$TABLELY_HOME/resume/<time>-<label>.md``)."""
    directory.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")[:40] or "resume"
    path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{safe}.md"
    path.write_text(prompt, encoding="utf-8")
    return path


def ai_argv(command: str, prompt: str, prompt_file: Path) -> List[str]:
    """How to start the AI CLI with a summary: as its last argument (``claude "<summary>"``),
    or as a file path where the command says ``{prompt_file}``."""
    parts = shlex.split(command)
    if any("{prompt_file}" in part for part in parts):
        return [part.replace("{prompt_file}", str(prompt_file)) for part in parts]
    return parts + [prompt]


def run_ai(command: str, prompt: str, prompt_file: Path, cwd: Optional[str] = None) -> int:
    """Start the AI CLI in this terminal with the summary and wait for it to finish."""
    try:
        return subprocess.call(ai_argv(command, prompt, prompt_file), cwd=cwd)
    except OSError as exc:
        raise RuntimeError(f"could not start {shlex.split(command)[0]!r}: {exc.strerror or exc}") from None
