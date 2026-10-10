"""Allocation logic: who gets which GPUs and CPU cores.

This module is pure — no processes, no I/O — so every decision can be tested
and previewed (``tablely plan``) without running anything.

One scheduling round:

1. Pending jobs are visited in priority order (ties: submission order).
   A job gets whole GPUs if it wants them and enough are free; a job asking
   for part of a GPU (``gpu_share`` / ``gpu_memory``) goes to the fullest GPU
   it still fits on (best fit), so small jobs pack together and whole GPUs
   stay free for big ones. A ``device = "any"`` job that cannot get a GPU runs
   on CPU instead; a ``device = "gpu"`` job waits.
2. Every job must also fit its guaranteed core count (``cpus``) and the
   RAM it reserves for tables (``ram``), when the inventory hands out RAM.
3. Strict priority (default): once a job has to wait, lower-priority jobs may
   not take what it is waiting for — free GPUs stay reserved for it and its
   cores and RAM are set aside — so important jobs are never starved. With
   ``backfill`` lower-priority jobs may use those resources in the meantime.
4. GPUs still free when nobody is waiting for one go to jobs starting now
   that accept more (``max_gpus``), in proportion to priority. A running job's
   GPU set never changes.
5. All cores are then split among the running jobs: everyone gets their
   minimum, and spare cores go out in proportion to priority (weighted
   max-min fairness), respecting each job's cap. Running jobs keep their GPUs
   but their core sets may grow or shrink; the cores they already hold are
   kept where possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .resources import Inventory, format_bytes
from .spec import Device, JobSpec

GPU = "gpu"
CPU = "cpu"
EPS = 1e-6  # slack when adding up GPU shares


@dataclass(frozen=True)
class Allocation:
    device: str  # where the job actually runs: "gpu" or "cpu"
    gpus: Tuple[str, ...] = ()
    cpus: Tuple[int, ...] = ()
    gpu_share: Optional[float] = None  # part of its one GPU, when it shares that GPU with others

    @property
    def on_gpu(self) -> bool:
        return self.device == GPU

    @property
    def gpu_load(self) -> float:
        """How much of each of its GPUs this job takes (1.0 = all of it)."""
        return 1.0 if self.gpu_share is None else self.gpu_share


@dataclass(frozen=True)
class Policy:
    backfill: bool = False


@dataclass
class Plan:
    allocations: Dict[str, Allocation]  # every job that should be running after this round
    started: List[str]  # newly admitted jobs, highest priority first
    waiting: Dict[str, str]  # pending job -> why it is not starting yet
    spare_gpus: List[str] = field(default_factory=list)  # free, and no waiting job wants them
    gpu_load: Dict[str, float] = field(default_factory=dict)  # GPU -> part in use after this round


def plan(
    inventory: Inventory,
    jobs: Mapping[str, JobSpec],
    running: Mapping[str, Allocation],
    pending: Sequence[str],
    policy: Policy = Policy(),
) -> Plan:
    """Decide which pending jobs start now and how cores are split.

    ``jobs`` holds every job in submission order; ``running`` the current
    allocation of jobs already started; ``pending`` the names still queued.
    """
    order = {name: i for i, name in enumerate(jobs)}

    def by_priority(name: str) -> Tuple[float, int]:
        return (-jobs[name].priority, order[name])

    total_cpus = len(inventory.cpus)
    used = gpu_load(inventory.gpus, running.values())
    free_gpus = [gpu for gpu in inventory.gpus if used[gpu] <= EPS]
    placed: Dict[str, Tuple[str, Tuple[str, ...], Optional[float]]] = {
        name: (alloc.device, alloc.gpus, alloc.gpu_share) for name, alloc in running.items()
    }
    committed = sum(jobs[name].cpus for name in running)
    ram_committed = sum(jobs[name].ram or 0 for name in running)

    started: List[str] = []
    waiting: Dict[str, str] = {}
    gpus_held_back = False  # a higher-priority job is waiting for GPUs
    cpus_held_back = False  # a higher-priority job is waiting for cores
    ram_held_back = False  # a higher-priority job is waiting for RAM
    for name in sorted(pending, key=by_priority):
        spec = jobs[name]
        slot = None
        if spec.device is not Device.CPU and not gpus_held_back:
            slot = fit_gpus(spec, inventory, used, free_gpus)
        if spec.device is Device.GPU and slot is None:
            if gpus_held_back:
                waiting[name] = "GPUs held for a higher-priority job"
            else:
                waiting[name] = _gpu_wait_reason(spec, inventory, used, free_gpus)
            if not policy.backfill:
                gpus_held_back = True
                committed += spec.cpus
                ram_committed += spec.ram or 0
            continue
        if cpus_held_back or committed + spec.cpus > total_cpus:
            if cpus_held_back:
                waiting[name] = "cores held for a higher-priority job"
            else:
                waiting[name] = f"needs {spec.cpus} core(s), {max(total_cpus - committed, 0)} free"
            if not policy.backfill:
                cpus_held_back = True
            continue
        if spec.ram and inventory.ram is not None and (
            ram_held_back or ram_committed + spec.ram > inventory.ram
        ):
            if ram_held_back:
                waiting[name] = "RAM held for a higher-priority job"
            else:
                free = format_bytes(max(inventory.ram - ram_committed, 0))
                waiting[name] = f"needs {format_bytes(spec.ram)} of RAM, {free} free"
            if not policy.backfill:
                ram_held_back = True
            continue

        if slot is not None:
            taken, share = slot
            for gpu in taken:
                used[gpu] += 1.0 if share is None else share
            free_gpus = [gpu for gpu in free_gpus if used[gpu] <= EPS]
            placed[name] = (GPU, taken, share)
        else:
            placed[name] = (CPU, (), None)
        committed += spec.cpus
        ram_committed += spec.ram or 0
        started.append(name)

    gpu_wanted = any(jobs[name].device is not Device.CPU for name in waiting)
    if free_gpus and not gpu_wanted:
        _grow_gpu_sets(jobs, started, placed, free_gpus, len(inventory.gpus))

    active = sorted(placed, key=by_priority)
    counts = share_cpus(
        total_cpus,
        [
            (jobs[name].cpus, jobs[name].cpu_cap(placed[name][0] == GPU), jobs[name].priority)
            for name in active
        ],
    )
    current = {name: alloc.cpus for name, alloc in running.items()}
    cpu_sets = _pick_cpu_ids(inventory.cpus, active, counts, current)
    allocations = {
        name: Allocation(device=placed[name][0], gpus=placed[name][1], cpus=cpu_sets[name],
                         gpu_share=placed[name][2])
        for name in active
    }
    spare = [] if gpu_wanted else list(free_gpus)
    return Plan(allocations=allocations, started=started, waiting=waiting, spare_gpus=spare,
                gpu_load=gpu_load(inventory.gpus, allocations.values()))


def gpu_load(gpus: Sequence[str], allocations: Iterable[Allocation]) -> Dict[str, float]:
    """How much of each GPU in ``gpus`` the given allocations take (0 = free, 1 = full)."""
    load = {gpu: 0.0 for gpu in gpus}
    for alloc in allocations:
        for gpu in alloc.gpus:
            if gpu in load:
                load[gpu] += alloc.gpu_load
    return load


def fit_gpus(
    spec: JobSpec, inventory: Inventory, used: Mapping[str, float], free_gpus: Sequence[str]
) -> Optional[Tuple[Tuple[str, ...], Optional[float]]]:
    """GPUs for ``spec`` right now as ``(gpus, share)``, or None if it does not fit.

    Whole-GPU jobs take free GPUs in order. A shared job takes the GPU it
    leaves the least room on (best fit), so partly used GPUs fill up first.
    """
    if not spec.shared:
        return (tuple(free_gpus[: spec.gpus]), None) if spec.gpus <= len(free_gpus) else None
    best: Optional[Tuple[float, str, float]] = None
    for gpu in inventory.gpus:
        share = spec.share_on(inventory.memory_of(gpu))
        if share is None:
            continue
        room = 1.0 - used[gpu] - share
        if room >= -EPS and (best is None or room < best[0] - EPS):
            best = (room, gpu, share)
    return None if best is None else ((best[1],), best[2])


def _gpu_wait_reason(
    spec: JobSpec, inventory: Inventory, used: Mapping[str, float], free_gpus: Sequence[str]
) -> str:
    if not spec.shared:
        return f"needs {spec.gpus} GPU(s), {len(free_gpus)} free"
    if spec.gpu_share is not None:
        most = max((1.0 - used[gpu] for gpu in inventory.gpus), default=0.0)
        return f"needs {spec.gpu_share:g} of a GPU, at most {max(most, 0.0):.2g} free on one"
    sizes = [(gpu, inventory.memory_of(gpu)) for gpu in inventory.gpus]
    room = [mem * max(1.0 - used[gpu], 0.0) for gpu, mem in sizes if mem]
    if not room:
        return f"needs {format_bytes(spec.gpu_memory)} of GPU memory; GPU memory sizes are unknown"
    return f"needs {format_bytes(spec.gpu_memory)} of GPU memory, at most {format_bytes(int(max(room)))} free on one"


def _grow_gpu_sets(
    jobs: Mapping[str, JobSpec],
    started: Sequence[str],
    placed: Dict[str, Tuple[str, Tuple[str, ...], Optional[float]]],
    free_gpus: List[str],
    total_gpus: int,
) -> None:
    """Hand GPUs nobody waits for to starting jobs that accept more, by priority."""
    growable = [
        name for name in started
        if placed[name][0] == GPU and placed[name][2] is None
        and jobs[name].gpu_cap(total_gpus) > len(placed[name][1])
    ]
    if not growable:
        return
    have = [len(placed[name][1]) for name in growable]
    counts = share_cpus(  # same weighted water-filling, applied to GPUs
        sum(have) + len(free_gpus),
        [(n, jobs[name].gpu_cap(total_gpus), jobs[name].priority) for name, n in zip(growable, have)],
    )
    for name, n, count in zip(growable, have, counts):
        extra = tuple(free_gpus[: count - n])
        del free_gpus[: count - n]
        placed[name] = (GPU, placed[name][1] + extra, None)


def share_cpus(total: int, demands: Sequence[Tuple[int, Optional[int], float]]) -> List[int]:
    """Split ``total`` units (cores, or GPUs) among ``(minimum, cap, weight)`` demands.

    Each demand gets its minimum, then spare units go one at a time to the
    demand with the fewest cores per unit of weight that is still below its
    cap. Ties favour earlier demands, so pass them highest priority first.
    Cores no one can take are left unassigned.
    """
    counts = [minimum for minimum, _, _ in demands]
    spare = total - sum(counts)
    if spare < 0:
        raise ValueError(f"minimum core demands ({sum(counts)}) exceed the {total} available")
    while spare > 0:
        best: Optional[int] = None
        best_score = 0.0
        for i, (_, cap, weight) in enumerate(demands):
            if cap is not None and counts[i] >= cap:
                continue
            score = counts[i] / weight
            if best is None or score < best_score:
                best, best_score = i, score
        if best is None:
            break
        counts[best] += 1
        spare -= 1
    return counts


def _pick_cpu_ids(
    all_cpus: Sequence[int],
    active: Sequence[str],
    counts: Sequence[int],
    current: Mapping[str, Tuple[int, ...]],
) -> Dict[str, Tuple[int, ...]]:
    """Turn core counts into concrete core ids, keeping running jobs where they are."""
    valid = set(all_cpus)
    chosen: Dict[str, List[int]] = {}
    used = set()
    for name, count in zip(active, counts):
        keep = [cpu for cpu in current.get(name, ()) if cpu in valid][:count]
        chosen[name] = keep
        used.update(keep)
    free = [cpu for cpu in all_cpus if cpu not in used]
    for name, count in zip(active, counts):
        need = count - len(chosen[name])
        chosen[name].extend(free[:need])
        del free[:need]
    return {name: tuple(sorted(cpus)) for name, cpus in chosen.items()}


def check_feasible(inventory: Inventory, jobs: Sequence[JobSpec]) -> Tuple[List[str], List[str]]:
    """Return ``(errors, warnings)`` for jobs that could never run as configured."""
    errors: List[str] = []
    warnings: List[str] = []
    seen = set()
    for spec in jobs:
        if spec.name in seen:
            errors.append(f"duplicate job name {spec.name!r}")
        seen.add(spec.name)
        if spec.ram and inventory.ram is not None and spec.ram > inventory.ram:
            errors.append(
                f"{spec.name}: reserves {format_bytes(spec.ram)} of RAM but only "
                f"{format_bytes(inventory.ram)} can be reserved ([resources] ram)"
            )
        if spec.cpus > len(inventory.cpus):
            errors.append(
                f"{spec.name}: needs {spec.cpus} core(s) but only {len(inventory.cpus)} are available"
            )
        if spec.device is Device.GPU and spec.gpus > len(inventory.gpus):
            errors.append(
                f"{spec.name}: needs {spec.gpus} GPU(s) but only {len(inventory.gpus)} are available"
            )
        elif spec.device is Device.ANY and spec.gpus > len(inventory.gpus):
            warnings.append(
                f"{spec.name}: wants {spec.gpus} GPU(s) but only {len(inventory.gpus)} exist; "
                "it will always run on CPU"
            )
        elif spec.gpu_memory is not None and inventory.gpus:
            sizes = [inventory.memory_of(gpu) for gpu in inventory.gpus]
            if not any(sizes):
                problem = (f"asks for {format_bytes(spec.gpu_memory)} of GPU memory but the GPUs' "
                           "memory size is unknown; set [resources] gpu_memory (or --gpu-memory)")
            elif spec.gpu_memory > max(m for m in sizes if m) * (1 + EPS):
                problem = (f"asks for {format_bytes(spec.gpu_memory)} of GPU memory but the largest "
                           f"GPU has {format_bytes(max(m for m in sizes if m))}")
            else:
                continue
            if spec.device is Device.GPU:
                errors.append(f"{spec.name}: {problem}")
            else:
                warnings.append(f"{spec.name}: {problem}; it will always run on CPU")
    return errors, warnings
