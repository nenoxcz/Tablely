"""Tablely: share one machine's GPUs and CPU cores among training jobs by priority."""

from .ledger import Ledger, MemoryLedger
from .planner import Allocation, Plan, Policy, plan, share_cpus
from .resources import Inventory, build_inventory, detect_cpus, detect_gpus
from .runner import JobState, Runner
from .spec import Device, JobSpec

__version__ = "0.1.0"

__all__ = [
    "Allocation",
    "Device",
    "Inventory",
    "JobSpec",
    "JobState",
    "Ledger",
    "MemoryLedger",
    "Plan",
    "Policy",
    "Runner",
    "build_inventory",
    "detect_cpus",
    "detect_gpus",
    "plan",
    "share_cpus",
    "__version__",
]
