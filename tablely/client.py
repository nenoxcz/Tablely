"""Optional helpers for training scripts launched by Tablely.

Nothing here is required — every value is also in plain environment variables
(``TABLELY_DEVICE``, ``TABLELY_GPUS``, ``TABLELY_NUM_CPUS`` ...) — but these
make the common cases one-liners::

    from tablely import client

    device = client.device()          # "cuda" or "cpu", as Tablely decided
    client.sync_torch_threads()       # call now and then: follows core changes
    client.progress(f"epoch {e}/{n}, val acc {acc:.3f}")   # shown in `tablely status`
"""

from __future__ import annotations

import os
import time
from typing import List, Optional


def is_managed() -> bool:
    """True when this process was started by Tablely."""
    return "TABLELY_JOB" in os.environ


def device(default: str = "cpu") -> str:
    """Where to train: ``"cuda"`` or ``"cpu"`` (``default`` outside Tablely)."""
    return os.environ.get("TABLELY_DEVICE", default)


def gpus() -> List[str]:
    """GPU ids assigned to this job (empty on CPU)."""
    value = os.environ.get("TABLELY_GPUS", "")
    return [g for g in value.split(",") if g]


def num_cpus() -> int:
    """Cores this job may use *right now*.

    Tablely may grow or shrink a running job's cores when other jobs start or
    finish; the live CPU affinity reflects that, the environment variable only
    shows the count at launch.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    value = os.environ.get("TABLELY_NUM_CPUS")
    return int(value) if value else (os.cpu_count() or 1)


def sync_torch_threads() -> int:
    """Match PyTorch's intra-op thread count to the current core allocation."""
    import torch

    n = num_cpus()
    if torch.get_num_threads() != n:
        torch.set_num_threads(n)
    return n


def agent() -> Optional[str]:
    """The agent that started this job (``--agent`` / ``$TABLELY_AGENT``)."""
    return os.environ.get("TABLELY_AGENT")


def task() -> Optional[str]:
    """What this job is for, as given in the job file or ``--task``."""
    return os.environ.get("TABLELY_TASK")


def progress(text: str) -> bool:
    """Tell other agents how this job is doing, e.g. ``"epoch 3/10, loss 0.41"``.

    It shows up next to the job in ``tablely status``. Each call takes the
    shared lock briefly, so call it once per epoch or every few minutes, not
    every step. Returns False (and does nothing) outside a ``tablely run``.
    """
    home = os.environ.get("TABLELY_HOME")
    key = os.environ.get("TABLELY_JOB_KEY")
    if not home or not key:
        return False
    from .ledger import Ledger

    with Ledger(home).locked() as board:
        job = board.job(key)
        if job is None:
            return False
        job["progress"] = " ".join(str(text).split())[:200]
        job["progress_at"] = time.time()
    return True


def note(text: str, who: Optional[str] = None) -> None:
    """Record what an agent is working on right now (``tablely status``, ``tablely history``)."""
    from .ledger import Ledger, default_agent, make_event

    who = who or default_agent()
    with Ledger(os.environ.get("TABLELY_HOME")).locked() as board:
        board.set_note(who, text)
        board.log(
            make_event(
                time.time(), "note", who, os.environ.get("TABLELY_RUN"), os.environ.get("TABLELY_JOB"),
                os.environ.get("TABLELY_TASK"), text,
            )
        )
