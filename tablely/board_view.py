"""Text views of the shared ledger: who is doing what now, and what happened."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import _fmt
from .ledger import RUNNING, allocation_from_json
from .progress import agent_achievement, collect_runs
from .resources import format_cpu_list
from .spec import format_priority

NOTE_MAX_AGE = 24 * 3600  # notes older than this are hidden from `status`


def render_status(
    data: Mapping[str, Any], now: Optional[float] = None, events: Sequence[Mapping[str, Any]] = ()
) -> str:
    """``events`` (the history) lets completion rates count jobs live runs already finished."""
    now = time.time() if now is None else now
    live = [r for r in collect_runs(data, events) if r.live]
    done_by_agent = {a: agent_achievement(r for r in live if r.agent == a) for a in {r.agent for r in live}}
    runs: Mapping[str, Dict[str, Any]] = data.get("runs", {})
    jobs: Mapping[str, Dict[str, Any]] = data.get("jobs", {})
    notes = {
        agent: note
        for agent, note in data.get("notes", {}).items()
        if now - note.get("at", 0) <= NOTE_MAX_AGE
    }
    lines: List[str] = []
    pool = data.get("pool")
    if pool:
        gpus = ",".join(pool["gpus"]) if pool["gpus"] else "none"
        mode = "backfill" if pool["backfill"] else "strict priority"
        lines.append(
            f"machine: {len(pool['cpus'])} CPU core(s) [{format_cpu_list(pool['cpus'])}], "
            f"{len(pool['gpus'])} GPU(s) [{gpus}]   policy: {mode}"
        )
    if not runs and not jobs:
        lines.append("nobody is running Tablely jobs on this machine")

    agents: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for run in sorted(runs.values(), key=lambda r: r.get("started_at", 0)):
        entry = agents.setdefault(run["agent"], {"runs": 0, "running": 0, "waiting": 0, "tasks": []})
        entry["runs"] += 1
        if run.get("task") and run["task"] not in entry["tasks"]:
            entry["tasks"].append(run["task"])
    for job in jobs.values():
        entry = agents.setdefault(job["agent"], {"runs": 0, "running": 0, "waiting": 0, "tasks": []})
        entry["running" if job["state"] == RUNNING else "waiting"] += 1
        if not entry["tasks"] and job.get("task"):
            entry["tasks"].append(job["task"])
    for agent in notes:
        agents.setdefault(agent, {"runs": 0, "running": 0, "waiting": 0, "tasks": []})

    if agents:
        rows = []
        for agent, entry in agents.items():
            note = notes.get(agent)
            note_text = f"{_clip(note['text'], 48)} ({_ago(now - note['at'])})" if note else "-"
            rows.append(
                [
                    agent,
                    f"{done_by_agent[agent]:.0%}" if agent in done_by_agent else "-",
                    str(entry["runs"]),
                    str(entry["running"]),
                    str(entry["waiting"]),
                    _clip("; ".join(entry["tasks"]) or "-", 40),
                    note_text,
                ]
            )
        lines += ["", "agents:", _fmt.table(["AGENT", "DONE", "RUNS", "RUNNING", "WAITING", "TASK", "NOTE"], rows)]

    if jobs:
        ordered = sorted(
            jobs.values(),
            key=lambda j: (j["state"] != RUNNING, -j["priority"], j["seq"]),
        )
        rows = []
        for job in ordered:
            alloc = allocation_from_json(job.get("allocation"))
            if job["state"] == RUNNING:
                state = "orphan" if job.get("orphan") else "running"
                since = job.get("started_at") or now
                if job.get("orphan"):
                    info = "its runner is gone; holds resources until it exits"
                elif job.get("progress"):
                    info = f"{job['progress']} ({_ago(now - (job.get('progress_at') or now))})"
                else:
                    info = ""
            else:
                state, since, info = "waiting", job.get("submitted_at") or now, job.get("wait_reason") or ""
                alloc = None
            frac = job.get("progress_frac") if job["state"] == RUNNING else None
            rows.append(
                [
                    job["agent"],
                    job["name"],
                    _clip(job.get("task") or "-", 28),
                    format_priority(job["priority"]),
                    state,
                    f"{frac:.0%}" if frac is not None else "-",
                    _fmt.placement(alloc),
                    _fmt.cores(alloc),
                    _fmt.duration(now - since),
                    _clip(info, 56),
                ]
            )
        lines += [
            "",
            "jobs:",
            _fmt.table(["AGENT", "JOB", "TASK", "PRIO", "STATE", "DONE", "ON", "CORES", "TIME", "INFO"], rows),
        ]
    return "\n".join(lines)


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
