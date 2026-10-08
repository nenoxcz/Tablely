"""Text views of the shared ledger: who is doing what now, and what happened."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import _fmt
from .ledger import RUNNING, allocation_from_json
from .resources import format_cpu_list
from .spec import format_priority

NOTE_MAX_AGE = 24 * 3600  # notes older than this are hidden from `status`


def render_status(data: Mapping[str, Any], now: Optional[float] = None) -> str:
    now = time.time() if now is None else now
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
                    str(entry["runs"]),
                    str(entry["running"]),
                    str(entry["waiting"]),
                    _clip("; ".join(entry["tasks"]) or "-", 40),
                    note_text,
                ]
            )
        lines += ["", "agents:", _fmt.table(["AGENT", "RUNS", "RUNNING", "WAITING", "TASK", "NOTE"], rows)]

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
            rows.append(
                [
                    job["agent"],
                    job["name"],
                    _clip(job.get("task") or "-", 28),
                    format_priority(job["priority"]),
                    state,
                    _fmt.placement(alloc),
                    _fmt.cores(alloc),
                    _fmt.duration(now - since),
                    _clip(info, 56),
                ]
            )
        lines += [
            "",
            "jobs:",
            _fmt.table(["AGENT", "JOB", "TASK", "PRIO", "STATE", "ON", "CORES", "TIME", "INFO"], rows),
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
