"""Text views of the shared ledger: who is doing what now, and what happened."""

from __future__ import annotations

import time
from typing import Any, List, Mapping, Optional, Sequence

from . import _fmt
from .ledger import RUNNING, allocation_from_json
from .progress import collect_runs
from .resources import format_cpu_list
from .spec import format_priority

NOTE_MAX_AGE = 24 * 3600  # notes older than this are hidden from `status`


def render_status(
    data: Mapping[str, Any],
    now: Optional[float] = None,
    events: Sequence[Mapping[str, Any]] = (),
    sessions: Optional[Sequence[Mapping[str, Any]]] = None,
    hours: float = 12.0,
    color: bool = False,
) -> str:
    """Progress view for the terminal: live runs with a bar per job, recently
    finished runs, coding sessions (when given) and the agents' notes.

    ``events`` (the history) lets live runs count the jobs they already
    finished; ``sessions`` are coding sessions from the work log.
    """
    now = time.time() if now is None else now
    runs = collect_runs(data, events)
    live = [r for r in runs if r.live]
    recent = [r for r in runs if not r.live and (r.ended or 0) >= now - hours * 3600]
    entries = {(j["run"], j["name"]): j for j in data.get("jobs", {}).values()}
    lines: List[str] = []

    pool = data.get("pool")
    if pool:
        gpus = ",".join(pool["gpus"]) if pool["gpus"] else "none"
        mode = "backfill" if pool["backfill"] else "strict priority"
        lines.append(f"machine  {len(pool['cpus'])} CPU core(s) [{format_cpu_list(pool['cpus'])}] · "
                     f"{len(pool['gpus'])} GPU(s) [{gpus}] · {mode}")
    if not live:
        lines.append("nobody is running Tablely jobs on this machine")
    for run in live:
        lines += [""] + _run_block(run, entries, now, color)

    if recent:
        lines += ["", f"recently finished (last {hours:g}h)"]
        heads = [f"{r.agent} · {r.task or r.run}" for r in recent]
        head_cells = min(max(_fmt.width(h) for h in heads), 48)
        for run, head in zip(recent, heads):
            lines.append(
                f"  {_fmt.pad(_fmt.clip(head, head_cells), head_cells)}  {_bar(run.achievement, 20, run, color)} "
                f"{_fmt.percent(run.achievement)}  {_counts(run, color)} · ended {_ago(now - run.ended)} · run {run.run}"
            )

    if sessions:
        lines += ["", "coding sessions"]
        heads = [f"{s['session'][:8]} · {s.get('task') or '-'}" for s in sessions]
        head_cells = min(max(_fmt.width(h) for h in heads), 48)
        for session, head in zip(sessions, heads):
            todos = session.get("todos") or []
            done = sum(1 for t in todos if t.get("status") == "completed")
            fraction = done / len(todos) if todos else None
            tasks = f"{done}/{len(todos)} tasks · " if todos else ""
            lines.append(
                f"  {_fmt.pad(_fmt.clip(head, head_cells), head_cells)}  {_fmt.bar(fraction, 20)} "
                f"{_fmt.percent(fraction)}  {tasks}{session.get('status', '?')} · {session.get('branch') or '?'}"
            )
            unfinished = [t for t in todos if t.get("status") != "completed"]
            if unfinished:
                lines.append(_fmt.paint(f"      next: {_fmt.clip(unfinished[0].get('subject', ''), 70)}", "dim", color))

    notes = sorted(
        ((agent, note) for agent, note in data.get("notes", {}).items() if now - note.get("at", 0) <= NOTE_MAX_AGE),
        key=lambda item: -item[1].get("at", 0),
    )
    if notes:
        lines += ["", "notes"]
        for agent, note in notes:
            lines.append(f"  {agent} ({_ago(now - note['at'])}): {_fmt.clip(note['text'], 100)}")

    if live or recent or sessions:
        lines += ["", _fmt.paint("resume any of these with: tablely resume", "dim", color)]
    return "\n".join(lines)


STATE_COLORS = {"ok": "green", "failed": "red", "cancelled": "dim", "running": "blue", "waiting": "dim"}


def _bar(fraction: Optional[float], cells: int, run: Any, color: bool) -> str:
    tint = "blue" if run.live else ("yellow" if run.count("failed") else "green")
    return _fmt.paint(_fmt.bar(fraction, cells), tint, color)


def _counts(run: Any, color: bool) -> str:
    parts = []
    for state in ("ok", "failed", "running", "waiting", "cancelled"):
        n = run.count(state)
        if n:
            parts.append(_fmt.paint(f"{n} {state}", STATE_COLORS[state], color))
    return " · ".join(parts) or "no jobs"


def _run_block(run: Any, entries: Mapping[Any, Mapping[str, Any]], now: float, color: bool) -> List[str]:
    since = f"running {_fmt.duration(now - run.started)}" if run.started else "running"
    head = f"{run.agent} · {run.task or run.run}"
    lines = [
        _fmt.paint(head, "bold", color) + _fmt.paint(f"   run {run.run} · {since}", "dim", color),
        f"  {_bar(run.achievement, 20, run, color)} {_fmt.percent(run.achievement)}   {_counts(run, color)}",
    ]
    name_cells = min(max((_fmt.width(j.name) for j in run.jobs), default=4), 24)
    for job in run.jobs:
        entry = entries.get((run.run, job.name)) or {}
        fraction = 1.0 if job.state == "ok" else job.fraction
        state = "orphan" if entry.get("orphan") else job.state
        where, info = "", job.detail or ""
        prio = f"prio {format_priority(entry['priority'])}" if "priority" in entry else ""
        if job.state == "running":
            alloc = allocation_from_json(entry.get("allocation"))
            where = " · ".join(x for x in (_fmt.placement(alloc), f"cores {_fmt.cores(alloc)}", prio) if x)
            info = job.progress or ""
            if entry.get("progress_at"):
                info += f" ({_ago(now - entry['progress_at'])})"
        elif job.state == "waiting":
            where = prio
        else:
            info = (job.progress or "") + (f" · {job.detail}" if job.state != "ok" and job.detail else "")
        lines.append(
            f"    {_fmt.pad(_fmt.clip(job.name, name_cells), name_cells)}  "
            f"{_fmt.paint(_fmt.bar(fraction if job.state != 'waiting' else None, 10), STATE_COLORS.get(job.state), color)} "
            f"{_fmt.percent(fraction if job.state != 'waiting' else None)}  "
            f"{_fmt.paint(_fmt.pad(state, 9), STATE_COLORS.get(job.state), color)}"
            f"{_fmt.pad(where, 34)}  {_fmt.clip(info, 60)}".rstrip()
        )
    return lines


def render_history(events: Sequence[Mapping[str, Any]]) -> str:
    if not events:
        return "no history yet"
    rows = []
    for event in events:
        rows.append(
            [
                str(event.get("time", "")).replace("T", " "),
                str(event.get("agent") or "-"),
                str(event.get("event") or "-"),
                str(event.get("job") or "-"),
                _clip(str(event.get("detail") or ""), 60),
                _clip(str(event.get("task") or ""), 32),
            ]
        )
    return _fmt.table(["TIME", "AGENT", "EVENT", "JOB", "DETAIL", "TASK"], rows, indent="")


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _ago(seconds: float) -> str:
    return f"{_fmt.duration(max(seconds, 0))} ago"


def render_brief(
    data: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    now: Optional[float] = None,
    hours: float = 24.0,
    agent: Optional[str] = None,
) -> str:
    """A handoff for whoever picks up next: what runs now, what finished, what agents said.

    Written to be pasted into an agent's context as-is.
    """
    now = time.time() if now is None else now
    since = now - hours * 3600

    def wanted(who: Optional[str]) -> bool:
        return agent is None or who == agent

    scope = f"last {hours:g}h" + (f", agent {agent}" if agent else "")
    lines = [f"# Tablely brief ({time.strftime('%Y-%m-%d %H:%M', time.localtime(now))}, {scope})"]
    pool = data.get("pool")
    if pool:
        mode = "backfill" if pool["backfill"] else "strict priority"
        lines.append(
            f"machine: {len(pool['cpus'])} core(s), GPUs [{','.join(pool['gpus']) or 'none'}], {mode}"
        )

    runs = [r for r in collect_runs(data, events) if wanted(r.agent) and (r.live or (r.ended or 0) >= since)]
    lines += ["", "## Progress"]
    if not runs:
        lines.append("- no runs in this window")
    for r in runs:
        counts = ", ".join(f"{r.count(s)} {s}" for s in ("ok", "failed", "running", "waiting") if r.count(s))
        lines.append(f"- [{r.agent}] {r.task or r.run}: {r.achievement:.0%} done ({counts or 'no jobs'})"
                     + ("" if r.live else ", ended"))

    lines += ["", "## Now"]
    jobs = [j for j in data.get("jobs", {}).values() if wanted(j["agent"])]
    if not jobs:
        lines.append("- nothing running or queued")
    for job in sorted(jobs, key=lambda j: (j["state"] != RUNNING, -j["priority"], j["seq"])):
        head = f"- [{job['agent']}] {job['name']}" + (f" ({job['task']})" if job.get("task") else "")
        if job["state"] == RUNNING:
            alloc = allocation_from_json(job.get("allocation"))
            body = (
                f"running on {_fmt.placement(alloc)}, cores {_fmt.cores(alloc)}, "
                f"{_fmt.duration(now - (job.get('started_at') or now))} so far"
            )
            if job.get("progress"):
                frac = job.get("progress_frac")
                body += f"; progress: {job['progress']}" + (f" ({frac:.0%})" if frac is not None else "")
            if job.get("orphan"):
                body += "; its runner is gone"
        else:
            body = f"waiting {_fmt.duration(now - (job.get('submitted_at') or now))}: {job.get('wait_reason') or '-'}"
        lines.append(f"{head}: {body}")

    recent = [e for e in events if e.get("t", 0) >= since and wanted(e.get("agent"))]
    finished = [e for e in recent if e.get("event") in ("done", "fail", "stop")][-15:][::-1]
    lines += ["", "## Finished (newest first)"]
    if not finished:
        lines.append("- nothing finished in this window")
    for e in finished:
        result = {"done": "ok", "stop": "cancelled"}.get(e["event"]) or e.get("detail") or "failed"
        line = f"- {_clock(e)} [{e.get('agent')}] {e.get('job')}" + (f" ({e['task']})" if e.get("task") else "")
        line += f": {result}"
        if e.get("seconds") is not None and e["event"] == "done":
            line += f" in {_fmt.duration(e['seconds'])}"
        if e.get("progress"):
            line += f"; last progress: {e['progress']}"
        git = e.get("git")
        if git:
            line += f"; code {git['commit']}{'+dirty' if git.get('dirty') else ''}"
            line += f" on {git['branch']}" if git.get("branch") else ""
        if e["event"] == "fail" and e.get("log"):
            line += f"; log {e['log']}"
        lines.append(line)

    notes = [e for e in recent if e.get("event") == "note"][-10:][::-1]
    lines += ["", "## Notes from agents (newest first)"]
    if not notes:
        lines.append("- none")
    for e in notes:
        lines.append(f"- {_clock(e)} [{e.get('agent')}] {e.get('detail')}")

    problems = [e for e in recent if e.get("event") in ("run-lost", "orphan-exit")]
    if problems:
        lines += ["", "## Problems"]
        for e in problems[-5:][::-1]:
            lines.append(f"- {_clock(e)} [{e.get('agent')}] {e.get('event')}: {e.get('detail')}")
    return "\n".join(lines)


def _clock(event: Mapping[str, Any]) -> str:
    stamp = str(event.get("time", ""))
    return stamp[11:16] if len(stamp) >= 16 else stamp
