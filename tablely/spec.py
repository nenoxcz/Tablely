"""Job specification: what a training job needs and how important it is."""

from __future__ import annotations

import enum
import math
import re
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union

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
        elif self.gpus < 1:
            raise ValueError(f"{self.name}: gpus must be at least 1 for device {self.device.value!r}")
        if self.max_gpus is not None and self.max_gpus != "all":
            if not _is_int(self.max_gpus):
                raise ValueError(f'{self.name}: max_gpus must be an integer or "all"')
            if self.max_gpus < self.gpus:
                raise ValueError(f"{self.name}: max_gpus ({self.max_gpus}) is below gpus ({self.gpus})")
        if not isinstance(self.switchable, bool):
            raise ValueError(f"{self.name}: switchable must be true or false")
        object.__setattr__(self, "env", {str(k): str(v) for k, v in dict(self.env).items()})
        if self.task is not None and not isinstance(self.task, str):
            raise ValueError(f"{self.name}: task must be a string")

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

