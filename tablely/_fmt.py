"""Small text-formatting helpers shared by the runner and the CLI."""

from __future__ import annotations

import os
import sys
import unicodedata
from typing import Optional, Sequence, TextIO

from .planner import Allocation
from .resources import format_cpu_list


def width(text: str) -> int:
    """Terminal cells ``text`` takes: Korean/CJK characters are two cells wide."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text: str, cells: int) -> str:
    return text + " " * max(cells - width(text), 0)


def clip(text: str, cells: int) -> str:
    """Shorten ``text`` (whitespace collapsed) to at most ``cells`` terminal cells."""
    text = " ".join(str(text).split())
    if width(text) <= cells:
        return text
    out, used = [], 0
    for ch in text:
        w = width(ch)
        if used + w > cells - 1:
            break
        out.append(ch)
        used += w
    return "".join(out) + "…"


def bar(fraction: Optional[float], cells: int = 20) -> str:
    """``████████░░░░`` for a fraction in 0..1 (all light when unknown)."""
    filled = 0 if fraction is None else round(max(0.0, min(1.0, fraction)) * cells)
    return "█" * filled + "░" * (cells - filled)


def percent(fraction: Optional[float]) -> str:
    return "   -" if fraction is None else f"{fraction:>4.0%}"


COLORS = {"green": "32", "red": "31", "yellow": "33", "blue": "34", "dim": "2", "bold": "1"}


def use_color(stream: Optional[TextIO] = None) -> bool:
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)()) and "NO_COLOR" not in os.environ


def paint(text: str, color: Optional[str], on: bool) -> str:
    return f"\033[{COLORS[color]}m{text}\033[0m" if on and color else text


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], indent: str = "  ") -> str:
    widths = [width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], width(cell))
    lines = []
    for row in [list(headers)] + [list(r) for r in rows]:
        cells = [pad(cell, widths[i]) for i, cell in enumerate(row)]
        lines.append((indent + "  ".join(cells)).rstrip())
    return "\n".join(lines)


def placement(alloc: Optional[Allocation]) -> str:
    """``GPU 0,1``, ``GPU 0 (share 0.5)`` for part of a shared GPU, or ``CPU``."""
    if alloc is None:
        return "-"
    if not alloc.on_gpu:
        return "CPU"
    return gpu_placement(alloc.gpus, alloc.gpu_share)


def gpu_placement(gpus: Sequence[str], share: Optional[float]) -> str:
    where = f"GPU {','.join(gpus)}"
    return where if share is None else f"{where} (share {share:.2g})"


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
