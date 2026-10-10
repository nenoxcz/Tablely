"""Job specification: what a training job needs and how important it is."""

from __future__ import annotations

import enum
import math
import re
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union

from .resources import format_bytes, parse_bytes

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class Device(str, enum.Enum):
    GPU = "gpu"  # must run on GPU(s); waits until enough are free
    CPU = "cpu"  # runs on CPU cores only; GPUs are hidden from it
    ANY = "any"  # prefers GPU, falls back to CPU when no GPU is free


@dataclass(frozen=True)
class JobSpec:
    """A single training job.

    ``priority`` is both the scheduling order (higher first) and the weight used
    to split spare CPU cores, so a priority-4 job gets ~2x the spare cores of a
    priority-2 job. ``cpus`` is the guaranteed minimum number of cores.
    ``max_cpus`` caps how many cores the job can grow to; when unset, a job
    placed on GPU stays at ``cpus`` and a job placed on CPU is uncapped.
    ``gpus`` is the GPU count a job needs; with ``max_gpus`` (a number or
    ``"all"``) it may be handed more at launch when no one else is waiting for
    a GPU. ``switchable`` jobs can checkpoint and be restarted on another
    device when Tablely asks (see ``client.switch_requested``).

    Small jobs can share one GPU: ``gpu_share`` asks for a fraction of a GPU
    (``0.5`` = half) and ``gpu_memory`` for an amount of GPU memory
    (``"10GiB"``), which Tablely turns into the fraction of whichever GPU the
    job lands on. Shared jobs are packed onto the fullest GPU they still fit.

    ``ram`` reserves RAM before the job starts, for tables kept outside GPU
    memory (see :mod:`tablely.tables`); a job waits until that much is free.
    """

    name: str
    command: Union[str, Sequence[str]]
    priority: float = 1.0
    device: Device = Device.GPU
    gpus: int = 1
    cpus: int = 1
    max_cpus: Optional[int] = None
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    shell: bool = False
    task: Optional[str] = None  # what this job is for, shown to other agents
    max_gpus: Optional[Union[int, str]] = None  # grow to this many GPUs at launch if free; "all" = no cap
    switchable: bool = False  # supports checkpoint + restart on another device
    gpu_share: Optional[float] = None  # fraction of one GPU (0 < x < 1); others may use the rest
    gpu_memory: Optional[Union[int, str]] = None  # GPU memory to set aside instead, e.g. "10GiB"
    ram: Optional[Union[int, str]] = None  # RAM reserved for this job's tables (tablely.tables), e.g. "32GiB"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.match(self.name):
            raise ValueError(
                f"invalid job name {self.name!r}: use letters, digits, '.', '_' or '-'"
            )
        try:
            object.__setattr__(self, "device", Device(self.device))
        except ValueError:
            raise ValueError(
                f"{self.name}: device must be one of gpu, cpu, any (got {self.device!r})"
            ) from None

        if isinstance(self.command, str):
            if not self.command.strip():
                raise ValueError(f"{self.name}: command is empty")
        else:
            command = tuple(self.command)
            if not command or not all(isinstance(part, str) for part in command):
                raise ValueError(f"{self.name}: command must be a string or a list of strings")
            if self.shell:
                raise ValueError(f"{self.name}: shell = true needs the command as a single string")
            object.__setattr__(self, "command", command)

        if not _is_number(self.priority) or not self.priority > 0 or math.isinf(self.priority):
            raise ValueError(f"{self.name}: priority must be a positive number (got {self.priority!r})")
        for attr in ("gpus", "cpus"):
            if not _is_int(getattr(self, attr)):
                raise ValueError(f"{self.name}: {attr} must be an integer")
        if self.cpus < 1:
            raise ValueError(f"{self.name}: cpus must be at least 1")
        if self.max_cpus is not None:
            if not _is_int(self.max_cpus):
                raise ValueError(f"{self.name}: max_cpus must be an integer")
            if self.max_cpus < self.cpus:
                raise ValueError(f"{self.name}: max_cpus ({self.max_cpus}) is below cpus ({self.cpus})")
        if self.device is Device.CPU:
            object.__setattr__(self, "gpus", 0)
            object.__setattr__(self, "max_gpus", None)
            object.__setattr__(self, "gpu_share", None)
            object.__setattr__(self, "gpu_memory", None)
        elif self.gpus < 1:
            raise ValueError(f"{self.name}: gpus must be at least 1 for device {self.device.value!r}")
        if self.max_gpus is not None and self.max_gpus != "all":
            if not _is_int(self.max_gpus):
                raise ValueError(f'{self.name}: max_gpus must be an integer or "all"')
            if self.max_gpus < self.gpus:
                raise ValueError(f"{self.name}: max_gpus ({self.max_gpus}) is below gpus ({self.gpus})")
        self._check_sharing()
        if self.ram is not None:
            try:
                object.__setattr__(self, "ram", parse_bytes(self.ram))
            except ValueError as exc:
                raise ValueError(f"{self.name}: ram: {exc}") from None
        if not isinstance(self.switchable, bool):
            raise ValueError(f"{self.name}: switchable must be true or false")
        object.__setattr__(self, "env", {str(k): str(v) for k, v in dict(self.env).items()})
        if self.task is not None and not isinstance(self.task, str):
            raise ValueError(f"{self.name}: task must be a string")

    def _check_sharing(self) -> None:
        if self.gpu_share is not None:
            if not _is_number(self.gpu_share) or not 0 < self.gpu_share < 1:
                raise ValueError(
                    f"{self.name}: gpu_share must be between 0 and 1, e.g. 0.5 for half a GPU "
                    f"(got {self.gpu_share!r}); leave it out to use whole GPUs"
                )
            object.__setattr__(self, "gpu_share", float(self.gpu_share))
        if self.gpu_memory is not None:
            try:
                object.__setattr__(self, "gpu_memory", parse_bytes(self.gpu_memory))
            except ValueError as exc:
                raise ValueError(f"{self.name}: gpu_memory: {exc}") from None
        if not self.shared:
            return
        if self.gpu_share is not None and self.gpu_memory is not None:
            raise ValueError(f"{self.name}: set gpu_share or gpu_memory, not both")
        if self.gpus != 1:
            raise ValueError(f"{self.name}: a job sharing a GPU uses exactly one (gpus = 1)")
        if self.max_gpus is not None:
            raise ValueError(f"{self.name}: max_gpus cannot be combined with gpu_share/gpu_memory")

    @property
    def shared(self) -> bool:
        """Whether this job asks for part of a GPU rather than whole GPUs."""
        return self.gpu_share is not None or self.gpu_memory is not None

    def share_on(self, gpu_memory: Optional[int]) -> Optional[float]:
        """Fraction of a GPU with ``gpu_memory`` bytes this job takes (1.0 for whole-GPU jobs).

        None when the job asks for memory but the GPU's size is unknown; above
        1.0 when the GPU is too small.
        """
        if self.gpu_share is not None:
            return self.gpu_share
        if self.gpu_memory is not None:
            return self.gpu_memory / gpu_memory if gpu_memory else None
        return 1.0

    def describe_gpu_need(self) -> str:
        """``"x2"`` (GPUs), ``"x0.5"`` (of one GPU), ``"10GiB"`` (of GPU memory), or ``""`` on CPU."""
        if self.gpu_memory is not None:
            return format_bytes(self.gpu_memory)
        if self.gpu_share is not None:
            return f"x{self.gpu_share:g}"
        return f"x{self.gpus}" if self.gpus else ""

    def gpu_cap(self, total_gpus: int) -> int:
        """Most GPUs this job can use on a machine with ``total_gpus``."""
        if self.max_gpus is None:
            return self.gpus
        if self.max_gpus == "all":
            return max(total_gpus, self.gpus)
        return self.max_gpus

    def cpu_cap(self, on_gpu: bool) -> Optional[int]:
        """Upper bound on cores for this job given where it was placed (None = no cap)."""
        if self.max_cpus is not None:
            return self.max_cpus
        return self.cpus if on_gpu else None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def format_priority(priority: float) -> str:
    return str(int(priority)) if float(priority).is_integer() else f"{priority:g}"

