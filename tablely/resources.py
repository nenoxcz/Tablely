"""Discovering the machine's CPU cores and GPUs."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

CpuSetting = Union[None, int, str, Sequence[int]]
GpuSetting = Union[None, int, str, Sequence[str]]


@dataclass(frozen=True)
class Inventory:
    """The resources Tablely may hand out: logical CPU ids and GPU ids.

    ``gpu_memory`` lines up with ``gpus`` (bytes, or None when unknown); it is
    what ``gpu_memory`` requests of shared-GPU jobs are measured against.
    ``ram`` is how much RAM jobs may reserve for tables (``ram`` in a job),
    or None when RAM is not handed out.
    """

    cpus: Tuple[int, ...]
    gpus: Tuple[str, ...]
    gpu_memory: Tuple[Optional[int], ...] = ()
    ram: Optional[int] = None

    def memory_of(self, gpu: str) -> Optional[int]:
        try:
            return self.gpu_memory[self.gpus.index(gpu)]
        except (ValueError, IndexError):
            return None

    def gpu_labels(self) -> List[str]:
        """GPU ids with their memory when known, e.g. ``["0 (24GiB)", "1"]``."""
        labels = []
        for gpu in self.gpus:
            memory = self.memory_of(gpu)
            labels.append(f"{gpu} ({format_bytes(memory)})" if memory else gpu)
        return labels

    def describe(self) -> str:
        return (
            f"{len(self.cpus)} CPU core(s) [{format_cpu_list(self.cpus)}], "
            f"{len(self.gpus)} GPU(s) [{', '.join(self.gpu_labels()) or 'none'}]"
            + (f", {format_bytes(self.ram)} RAM for tables" if self.ram else "")
        )


_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(?:i?B)?\s*$", re.IGNORECASE)


def parse_bytes(value: Union[int, str]) -> int:
    """``"10GiB"``, ``"10G"``, ``"10GB"`` (all 1024-based), ``"512MiB"`` or a plain byte count."""
    if isinstance(value, bool):
        raise ValueError(f"not a memory size: {value!r}")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("memory size must be positive")
        return value
    match = _SIZE.match(str(value))
    if not match:
        raise ValueError(f"not a memory size: {value!r} (use e.g. \"10GiB\" or \"512MiB\")")
    number = float(match.group(1)) * 1024 ** "_KMGT".index(match.group(2).upper() or "_")
    if number <= 0:
        raise ValueError("memory size must be positive")
    return int(number)


def format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "?"
    for unit, size in (("TiB", 1024 ** 4), ("GiB", 1024 ** 3), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if value >= size:
            return f"{value / size:.3g}{unit}"
    return f"{value}B"


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


def total_ram_bytes() -> Optional[int]:
    """Physical RAM of this machine (MemTotal on Linux), or None if unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def detect_gpus(env: Optional[Mapping[str, str]] = None) -> List[str]:
    """GPU ids usable by jobs (see :func:`detect_gpu_devices`)."""
    return [gpu for gpu, _ in detect_gpu_devices(env)]


def detect_gpu_devices(env: Optional[Mapping[str, str]] = None) -> List[Tuple[str, Optional[int]]]:
    """``(id, memory in bytes or None)`` for every GPU jobs may use.

    If ``CUDA_VISIBLE_DEVICES`` is already set for Tablely itself, only those
    devices are used. Otherwise ``nvidia-smi`` is asked; no NVIDIA driver means
    no GPUs. A GPU split with MIG is replaced by its MIG instances (ids
    ``MIG-<uuid>``), each scheduled like a GPU of its own.
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
        memory = _nvidia_smi_memory() if gpus else {}
        return [(gpu, memory.get(gpu)) for gpu in gpus]
    listing = _nvidia_smi("-L")
    if listing is None:
        return []
    return parse_gpu_listing(listing, _nvidia_smi_memory())


def parse_gpu_listing(listing: str, memory: Mapping[str, int]) -> List[Tuple[str, Optional[int]]]:
    """Read ``nvidia-smi -L``: physical GPUs by index, MIG instances by UUID with their profile's memory."""
    devices: List[Tuple[str, Optional[int]]] = []
    current: Optional[str] = None
    migs: List[Tuple[str, Optional[int]]] = []

    def flush() -> None:
        if current is not None:
            devices.extend(migs or [(current, memory.get(current))])

    for line in listing.splitlines():
        gpu = re.match(r"^GPU (\d+):", line)
        mig = re.match(r"^\s+MIG\s+(\S+)\s+Device\s+\d+:\s*\(UUID:\s*(MIG-[^)\s]+)\)", line)
        if gpu:
            flush()
            current, migs = gpu.group(1), []
        elif mig and current is not None:
            size = re.search(r"(\d+)gb", mig.group(1), re.IGNORECASE)  # profile like "3g.40gb"
            migs.append((mig.group(2), int(size.group(1)) * 1024 ** 3 if size else None))
    flush()
    return devices


def _nvidia_smi(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=15, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def _nvidia_smi_memory() -> Dict[str, int]:
    """Total memory per GPU index, in bytes."""
    out = _nvidia_smi("--query-gpu=index,memory.total", "--format=csv,noheader,nounits") or ""
    memory = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            memory[parts[0]] = int(parts[1]) * 1024 ** 2  # reported in MiB
    return memory


def build_inventory(
    cpus: CpuSetting = None,
    gpus: GpuSetting = None,
    reserve_cpus: int = 0,
    simulate: bool = False,
    gpu_memory: Union[None, int, str] = None,
    ram: Union[None, int, str] = None,
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
    ``gpu_memory``: memory of each GPU (e.g. ``"24GiB"``), when it cannot be
    detected or should be overridden.
    ``ram``: RAM jobs may reserve for tables (e.g. ``"200GiB"``); default half
    of the machine's RAM, since reserved RAM is pinned and cannot be swapped.
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

    found = dict(detect_gpu_devices()) if gpus is None or isinstance(gpus, int) else {}
    if gpus is None:
        gpu_ids = list(found)
    elif isinstance(gpus, int) and not isinstance(gpus, bool):
        if gpus < 0:
            raise ValueError("gpus must not be negative")
        gpu_ids = list(found)[:gpus] if len(found) >= gpus else [str(i) for i in range(gpus)]
    elif isinstance(gpus, str):
        gpu_ids = [g.strip() for g in gpus.split(",") if g.strip()]
    else:
        gpu_ids = [str(g) for g in gpus]
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"duplicate GPU ids in {gpu_ids}")
    if gpu_memory is not None:
        memory = [parse_bytes(gpu_memory)] * len(gpu_ids)
    else:
        if gpu_ids and not found:
            found = dict(detect_gpu_devices())
        memory = [found.get(gpu) for gpu in gpu_ids]

    if ram is not None:
        ram_bytes: Optional[int] = parse_bytes(ram)
    else:
        total = total_ram_bytes()
        ram_bytes = total // 2 if total else None

    return Inventory(cpus=tuple(cpu_ids), gpus=tuple(gpu_ids), gpu_memory=tuple(memory), ram=ram_bytes)
