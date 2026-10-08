"""Optional helpers for training scripts launched by Tablely.

Nothing here is required — every value is also in plain environment variables
(``TABLELY_DEVICE``, ``TABLELY_GPUS``, ``TABLELY_NUM_CPUS`` ...) — but these
make the common cases one-liners::

    from tablely import client

    device = client.device()          # "cuda" or "cpu", as Tablely decided
    client.sync_torch_threads()       # call now and then: follows core changes
"""

from __future__ import annotations

import os
from typing import List


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
