"""Pinning job processes to CPU cores (Linux only; a no-op elsewhere)."""

from __future__ import annotations

import os
from typing import Callable, Iterator, Optional, Sequence

SUPPORTED = hasattr(os, "sched_setaffinity")


def pin_self_hook(cpus: Sequence[int]) -> Optional[Callable[[], None]]:
    """A ``preexec_fn`` that pins the child to ``cpus`` before it execs."""
    if not SUPPORTED or not cpus:
        return None
    mask = set(cpus)

    def hook() -> None:
        os.sched_setaffinity(0, mask)

    return hook


def pin_group(pgid: int, cpus: Sequence[int]) -> int:
    """Re-pin every thread of every process in process group ``pgid``.

    Changing a running job's cores has to reach all of its threads and child
    processes (e.g. data-loader workers), not just the main thread. Returns the
    number of threads updated.
    """
    if not SUPPORTED or not cpus:
        return 0
    mask = set(cpus)
    updated = 0
    for pid in _pids_in_group(pgid):
        for tid in _threads(pid):
            try:
                os.sched_setaffinity(tid, mask)
                updated += 1
            except OSError:  # thread exited meanwhile, or not ours to change
                pass
    return updated


def _pids_in_group(pgid: int) -> Iterator[int]:
    try:
        entries = os.listdir("/proc")
    except OSError:
        return
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            if os.getpgid(pid) == pgid:
                yield pid
        except OSError:
            continue


def _threads(pid: int) -> Iterator[int]:
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return
    for tid in tids:
        if tid.isdigit():
            yield int(tid)
