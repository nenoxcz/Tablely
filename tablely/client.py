"""Optional helpers for training scripts launched by Tablely.

Nothing here is required — every value is also in plain environment variables
(``TABLELY_DEVICE``, ``TABLELY_GPUS``, ``TABLELY_NUM_CPUS`` ...) — but these
make the common cases one-liners::

    from tablely import client

    device = client.device()          # "cuda" or "cpu", as Tablely decided
    client.sync_torch_threads()       # call now and then: follows core changes
    client.progress(f"epoch {e}/{n}, val acc {acc:.3f}")   # shown in `tablely status`

Sharing a GPU (jobs with ``gpu_share`` or ``gpu_memory``)::

    client.limit_gpu_memory()         # PyTorch: stay within this job's part of the GPU

Moving between CPU and GPU (jobs marked ``switchable = true``)::

    start = load_checkpoint() if client.restarts() else 0
    for epoch in range(start, epochs):
        train_one_epoch()
        if client.switch_requested():      # "gpu": one is free / "cpu": a bigger job needs it
            save_checkpoint(epoch + 1)
            client.exit_for_switch()       # Tablely restarts this job on the other device
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List, NoReturn, Optional

SWITCH_EXIT = 75  # must match tablely.runner.SWITCH_EXIT


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


def gpu_share() -> Optional[float]:
    """Part of its GPU this job was given (e.g. ``0.5``), or None when it has whole GPUs."""
    value = os.environ.get("TABLELY_GPU_SHARE")
    try:
        return float(value) if value and device() == "cuda" else None
    except ValueError:
        return None


def gpu_memory() -> Optional[int]:
    """GPU memory this job may use, in bytes, when it shares a GPU and the size is known."""
    value = os.environ.get("TABLELY_GPU_MEMORY")
    return int(value) if value and value.isdigit() and gpu_share() is not None else None


def limit_gpu_memory() -> Optional[float]:
    """Keep PyTorch within this job's share of the GPU; call once before training.

    Jobs sharing a GPU are trusted to stay within their part: JAX and
    TensorFlow are configured through environment variables, PyTorch needs
    this call (``torch.cuda.set_per_process_memory_fraction``). Allocations
    beyond the share then fail with an out-of-memory error in this job
    instead of crashing its neighbours. Does nothing for whole-GPU jobs and
    returns the fraction applied (or None).
    """
    share = gpu_share()
    if share is None:
        return None
    import torch

    if not torch.cuda.is_available():
        return None
    torch.cuda.set_per_process_memory_fraction(share, 0)
    return share


def ram() -> Optional[int]:
    """Bytes of RAM Tablely reserved for this job's tables (``ram`` in the job file), or None."""
    value = os.environ.get("TABLELY_RAM")
    return int(value) if value and value.isdigit() else None


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


def progress(
    text: Optional[str] = None,
    *,
    done: Optional[float] = None,
    total: Optional[float] = None,
    fraction: Optional[float] = None,
) -> bool:
    """Tell other agents how far this job is, e.g. ``progress("epoch 3/10, loss 0.41")``.

    The completion rate comes from ``fraction`` (0..1), or ``done``/``total``,
    or else is read from the text (``"3/10"``, ``"45%"``). It shows up as a
    percentage in ``tablely status`` and the dashboard. Each call takes the
    shared lock briefly, so call it once per epoch or every few minutes, not
    every step. Returns False (and does nothing) outside a ``tablely run``.
    """
    from .ledger import parse_fraction

    if fraction is None and done is not None and total:
        fraction = done / total
    if fraction is None:
        fraction = parse_fraction(text)
    if fraction is not None:
        fraction = min(max(float(fraction), 0.0), 1.0)
    if text is None:
        text = f"{done:g}/{total:g}" if done is not None and total else (
            f"{fraction:.0%}" if fraction is not None else "")
    return _update_job({
        "progress": " ".join(str(text).split())[:200],
        "progress_frac": fraction,
        "progress_at": time.time(),
    })


def _update_job(fields: dict, event: Optional[str] = None, detail: str = "") -> bool:
    """Set ``fields`` on this job's ledger entry (and log ``event`` to the history).

    Returns False, doing nothing, outside a ``tablely run``.
    """
    home = os.environ.get("TABLELY_HOME")
    key = os.environ.get("TABLELY_JOB_KEY")
    if not home or not key:
        return False
    from .ledger import Ledger, make_event

    with Ledger(home).locked() as board:
        job = board.job(key)
        if job is None:
            return False
        job.update(fields)
        if event:
            board.log(make_event(time.time(), event, job.get("agent"), job.get("run"), job.get("name"),
                                 job.get("task"), detail))
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


def restarts() -> int:
    """How many times this job was restarted to change devices (0 on the first run)."""
    try:
        return int(os.environ.get("TABLELY_RESTARTS", "0"))
    except ValueError:
        return 0


def switch_requested() -> Optional[str]:
    """``"gpu"`` or ``"cpu"`` when Tablely wants this job moved, else None.

    Cheap (one small file read), so it can be checked every epoch or every few
    hundred steps. Only asked of jobs marked ``switchable = true``.
    """
    path = os.environ.get("TABLELY_CONTROL")
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            target = json.load(f).get("switch_to")
    except (OSError, ValueError, AttributeError):
        return None
    return target if target in ("gpu", "cpu") else None


def exit_for_switch(want: Optional[str] = None) -> NoReturn:
    """End this run so Tablely restarts the job, after you saved a checkpoint.

    Without ``want`` the job goes wherever Tablely asked (``switch_requested``).
    ``want="cpu"`` keeps it on CPU from now on, e.g. after CUDA ran out of
    memory. Outside a switchable Tablely job this is a plain ``exit(75)``.
    """
    reply = os.environ.get("TABLELY_REPLY")
    if want is not None:
        if want not in ("gpu", "cpu"):
            raise ValueError('want must be "gpu" or "cpu"')
        if reply:
            with open(reply, "w", encoding="utf-8") as f:
                json.dump({"want": want, "at": time.time()}, f)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(SWITCH_EXIT)
