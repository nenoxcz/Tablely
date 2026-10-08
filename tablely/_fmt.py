"""Small text-formatting helpers shared by the runner and the CLI."""

from __future__ import annotations

from typing import Optional, Sequence

from .planner import Allocation
from .resources import format_cpu_list


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], indent: str = "  ") -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = []
    for row in [list(headers)] + [list(r) for r in rows]:
        cells = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
        lines.append((indent + "  ".join(cells)).rstrip())
    return "\n".join(lines)


def placement(alloc: Optional[Allocation]) -> str:
    if alloc is None:
        return "-"
    return f"GPU {','.join(alloc.gpus)}" if alloc.on_gpu else "CPU"


def cores(alloc: Optional[Allocation]) -> str:
    if alloc is None or not alloc.cpus:
        return "-"
    return f"{format_cpu_list(alloc.cpus)} ({len(alloc.cpus)})"


def duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
