"""Discovering the machine's CPU cores and GPUs."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

CpuSetting = Union[None, int, str, Sequence[int]]
GpuSetting = Union[None, int, str, Sequence[str]]


@dataclass(frozen=True)
class Inventory:
    """The resources Tablely may hand out: logical CPU ids and GPU ids."""

    cpus: Tuple[int, ...]
    gpus: Tuple[str, ...]

    def describe(self) -> str:
        gpus = ",".join(self.gpus) if self.gpus else "none"
        return (
            f"{len(self.cpus)} CPU core(s) [{format_cpu_list(self.cpus)}], "
            f"{len(self.gpus)} GPU(s) [{gpus}]"
        )


def parse_cpu_list(text: str) -> List[int]:
    """Parse the Linux cpulist format, e.g. ``"0-3,8,10-11"``."""
    cpus = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                lo_text, hi_text = part.split("-", 1)
                lo, hi = int(lo_text), int(hi_text)
                if lo < 0 or hi < lo:
                    raise ValueError
                cpus.update(range(lo, hi + 1))
            else:
                cpu = int(part)
                if cpu < 0:
                    raise ValueError
                cpus.add(cpu)
        except ValueError:
            raise ValueError(f"invalid CPU list entry {part!r} in {text!r}") from None
    return sorted(cpus)


def format_cpu_list(cpus: Iterable[int]) -> str:
    """Inverse of :func:`parse_cpu_list`: ``[0, 1, 2, 3, 8]`` -> ``"0-3,8"``."""
    ordered = sorted(set(cpus))
    ranges = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        ranges.append(str(ordered[i]) if i == j else f"{ordered[i]}-{ordered[j]}")
        i = j + 1
    return ",".join(ranges)


def detect_cpus() -> List[int]:
    """CPUs this process may run on (respects taskset/cgroup cpusets on Linux)."""
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def detect_gpus(env: Optional[Mapping[str, str]] = None) -> List[str]:
    """GPU ids usable by jobs.

    If ``CUDA_VISIBLE_DEVICES`` is already set for Tablely itself, only those
    devices are used. Otherwise ``nvidia-smi`` is asked; no NVIDIA driver means
    no GPUs.
    """
    env = os.environ if env is None else env
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        gpus = []
        for item in visible.split(","):
            item = item.strip()
            if not item or item.startswith("-"):  # CUDA stops at the first invalid id
                break
            gpus.append(item)
        return gpus
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def build_inventory(
    cpus: CpuSetting = None,
    gpus: GpuSetting = None,
    reserve_cpus: int = 0,
    simulate: bool = False,
) -> Inventory:
    """Build the inventory, optionally narrowing what was detected.

    ``cpus``: None = all detected; an int = the first N detected cores; a string
    such as ``"0-7,16-23"`` or a list of ids = exactly those cores.
    ``gpus``: None = detected; an int = the first N detected GPUs (ids
    ``0..N-1`` are assumed when fewer are detected); a string ``"0,1"`` or a
    list = exactly those ids.
    ``reserve_cpus``: keep the lowest N cores free for the OS and Tablely.
    ``simulate``: allow cores this machine does not have (for dry-run previews
    of a bigger server); never use it for real runs.
    """
    detected = detect_cpus()
    if cpus is None:
        cpu_ids = detected
    elif isinstance(cpus, int) and not isinstance(cpus, bool):
        if cpus < 1:
            raise ValueError("cpus must be at least 1")
        if cpus <= len(detected):
            cpu_ids = detected[:cpus]
        elif simulate:
            cpu_ids = list(range(cpus))
        else:
            raise ValueError(f"cpus = {cpus}, but {len(detected)} core(s) are available")
    else:
        cpu_ids = parse_cpu_list(cpus) if isinstance(cpus, str) else sorted(set(int(c) for c in cpus))
        if not cpu_ids:
            raise ValueError("the CPU list is empty")
        missing = sorted(set(cpu_ids) - set(detected))
        if missing and not simulate:
            raise ValueError(
                f"CPU(s) {format_cpu_list(missing)} are not available to this process "
                f"(available: {format_cpu_list(detected)})"
            )

    if isinstance(reserve_cpus, bool) or not isinstance(reserve_cpus, int) or reserve_cpus < 0:
        raise ValueError("reserve_cpus must be a non-negative integer")
    if reserve_cpus >= len(cpu_ids):
        raise ValueError(f"reserve_cpus = {reserve_cpus} leaves no cores for jobs")
    cpu_ids = cpu_ids[reserve_cpus:]

    if gpus is None:
        gpu_ids = detect_gpus()
    elif isinstance(gpus, int) and not isinstance(gpus, bool):
        if gpus < 0:
            raise ValueError("gpus must not be negative")
        found = detect_gpus()
        gpu_ids = found[:gpus] if len(found) >= gpus else [str(i) for i in range(gpus)]
    elif isinstance(gpus, str):
        gpu_ids = [g.strip() for g in gpus.split(",") if g.strip()]
    else:
        gpu_ids = [str(g) for g in gpus]
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"duplicate GPU ids in {gpu_ids}")

    return Inventory(cpus=tuple(cpu_ids), gpus=tuple(gpu_ids))
