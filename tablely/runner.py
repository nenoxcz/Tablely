"""Runs jobs as subprocesses according to the planner's decisions."""

from __future__ import annotations

import enum
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, TextIO, Union

from . import _fmt, affinity
from .planner import Allocation, Policy, check_feasible, plan
from .resources import Inventory, format_cpu_list
from .spec import JobSpec, format_priority

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_PLACEHOLDER = re.compile(r"\{(name|device|gpus|num_gpus|cpus|cpu_list)\}")


class JobState(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobRecord:
    spec: JobSpec
    state: JobState = JobState.PENDING
    allocation: Optional[Allocation] = None
    process: Optional[subprocess.Popen] = None
    log_path: Optional[Path] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    returncode: Optional[int] = None
    wait_reason: Optional[str] = None
    error: Optional[str] = None


def job_values(spec: JobSpec, alloc: Allocation) -> Dict[str, str]:
    """Values for command placeholders, as of launch time."""
    return {
        "name": spec.name,
        "device": "cuda" if alloc.on_gpu else "cpu",
        "gpus": ",".join(alloc.gpus),
        "num_gpus": str(len(alloc.gpus)),
        "cpus": str(len(alloc.cpus)),
        "cpu_list": format_cpu_list(alloc.cpus),
    }


def render_command(spec: JobSpec, alloc: Allocation) -> Union[str, List[str]]:
    """Fill ``{device}``, ``{gpus}``, ``{num_gpus}``, ``{cpus}``, ``{cpu_list}``, ``{name}``."""
    values = job_values(spec, alloc)

    def fill(text: str) -> str:
        return _PLACEHOLDER.sub(lambda m: values[m.group(1)], text)

    if isinstance(spec.command, str):
        command = fill(spec.command)
        return command if spec.shell else shlex.split(command)
    return [fill(part) for part in spec.command]


def build_env(base: Mapping[str, str], spec: JobSpec, alloc: Allocation) -> Dict[str, str]:
    """Environment for a job: its own ``env`` plus the resources it was given.

    GPU visibility is always enforced. Thread-count variables follow the core
    count unless the job sets them itself.
    """
    env = dict(base)
    env.update(spec.env)
    values = job_values(spec, alloc)
    for var in THREAD_ENV_VARS:
        if var not in spec.env:
            env[var] = values["cpus"]
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # match nvidia-smi numbering
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": values["gpus"],
            "TABLELY_JOB": spec.name,
            "TABLELY_PRIORITY": format_priority(spec.priority),
            "TABLELY_DEVICE": values["device"],
            "TABLELY_GPUS": values["gpus"],
            "TABLELY_NUM_CPUS": values["cpus"],
            "TABLELY_CPU_LIST": values["cpu_list"],
        }
    )
    return env


class Runner:
    """Runs a batch of jobs to completion, re-planning whenever one finishes."""

    def __init__(
        self,
        inventory: Inventory,
        jobs: Sequence[JobSpec],
        *,
        policy: Policy = Policy(),
        log_dir: Union[str, Path] = "tablely-logs",
        poll_interval: float = 0.5,
        stop_timeout: float = 10.0,
        out: Optional[TextIO] = None,
        base_env: Optional[Mapping[str, str]] = None,
    ) -> None:
        errors, self.warnings = check_feasible(inventory, jobs)
        if errors:
            raise ValueError("\n".join(errors))
        self.inventory = inventory
        self.policy = policy
        self.log_dir = Path(log_dir)
        self.poll_interval = poll_interval
        self.stop_timeout = stop_timeout
        self.out = out if out is not None else sys.stdout
        self.base_env = dict(os.environ if base_env is None else base_env)
        self.specs: Dict[str, JobSpec] = {spec.name: spec for spec in jobs}
        self.records: Dict[str, JobRecord] = {spec.name: JobRecord(spec) for spec in jobs}
        self._name_width = max((len(name) for name in self.specs), default=4)

    # -- public ---------------------------------------------------------------

    def run(self) -> int:
        """Run every job. Returns 0 if all succeeded, 1 if any failed, 130 if interrupted."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._say(f"resources: {self.inventory.describe()}")
        self._say(f"policy: {'backfill' if self.policy.backfill else 'strict priority'}, logs in {self.log_dir}")
        for warning in self.warnings:
            self._say(f"warning: {warning}")
        interrupted = False
        try:
            self._schedule()
            while self._in_state(JobState.RUNNING):
                if self._reap():
                    self._schedule()
                else:
                    time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            interrupted = True
            self._say("interrupted: stopping running jobs")
        finally:
            self._stop_all()
            self._say("")
            self._say(self.summary())
        if interrupted:
            return 130
        return 0 if all(r.state is JobState.SUCCEEDED for r in self.records.values()) else 1

    def summary(self) -> str:
        rows = []
        for rec in self.records.values():
            if rec.state is JobState.SUCCEEDED:
                result = "ok"
            elif rec.state is JobState.FAILED:
                result = rec.error or f"exit {rec.returncode}"
            else:
                result = rec.state.value
            took = (
                _fmt.duration(rec.ended_at - rec.started_at)
                if rec.started_at is not None and rec.ended_at is not None
                else "-"
            )
            rows.append(
                [rec.spec.name, format_priority(rec.spec.priority), _fmt.placement(rec.allocation), result, took]
            )
        return "summary:\n" + _fmt.table(["JOB", "PRIO", "RAN ON", "RESULT", "TIME"], rows)

    # -- scheduling -----------------------------------------------------------

    def _schedule(self) -> None:
        while True:
            running = {n: r.allocation for n, r in self.records.items() if r.state is JobState.RUNNING}
            pending = [n for n, r in self.records.items() if r.state is JobState.PENDING]
            decision = plan(self.inventory, self.specs, running, pending, self.policy)

            # Shrink/grow running jobs before starting new ones on the freed cores.
            for name, alloc in decision.allocations.items():
                rec = self.records[name]
                if rec.state is JobState.RUNNING and alloc.cpus != rec.allocation.cpus:
                    self._resize(rec, alloc.cpus)

            launch_failed = False
            for name in decision.started:
                if not self._launch(self.records[name], decision.allocations[name]):
                    launch_failed = True

            for name, reason in decision.waiting.items():
                rec = self.records[name]
                if rec.state is JobState.PENDING and reason != rec.wait_reason:
                    rec.wait_reason = reason
                    self._event("wait", rec, reason)

            if not launch_failed:
                return  # otherwise re-plan: the failed job's resources are free again

    def _launch(self, rec: JobRecord, alloc: Allocation) -> bool:
        spec = rec.spec
        rec.log_path = self.log_dir / f"{spec.name}.log"
        try:
            with open(rec.log_path, "wb") as log:
                rec.process = subprocess.Popen(
                    render_command(spec, alloc),
                    shell=spec.shell,
                    cwd=spec.cwd,
                    env=build_env(self.base_env, spec, alloc),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # own process group: lets us pin and stop the whole tree
                    preexec_fn=affinity.pin_self_hook(alloc.cpus),
                )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            rec.state = JobState.FAILED
            rec.error = f"launch failed: {exc}"
            rec.started_at = rec.ended_at = time.time()
            self._event("fail", rec, rec.error)
            return False
        rec.state = JobState.RUNNING
        rec.allocation = alloc
        rec.started_at = time.time()
        rec.wait_reason = None
        self._event(
            "start",
            rec,
            f"prio {format_priority(spec.priority)}  {_fmt.placement(alloc)}  cores {_fmt.cores(alloc)}",
        )
        return True

    def _resize(self, rec: JobRecord, cpus: Sequence[int]) -> None:
        old = rec.allocation
        rec.allocation = Allocation(device=old.device, gpus=old.gpus, cpus=tuple(cpus))
        affinity.pin_group(rec.process.pid, cpus)
        self._event("resize", rec, f"cores {_fmt.cores(old)} -> {_fmt.cores(rec.allocation)}")

    def _reap(self) -> bool:
        finished = False
        for rec in self.records.values():
            if rec.state is not JobState.RUNNING:
                continue
            if not _has_exited(rec.process):
                continue
            code = self._finish(rec)
            rec.state = JobState.SUCCEEDED if code == 0 else JobState.FAILED
            took = _fmt.duration(rec.ended_at - rec.started_at)
            if code == 0:
                self._event("done", rec, f"ok in {took}")
            else:
                self._event("fail", rec, f"exit {code} after {took}, see {rec.log_path}")
            finished = True
        return finished

    def _stop_all(self) -> None:
        running = self._in_state(JobState.RUNNING)
        for rec in running:
            self._signal_group(rec, signal.SIGTERM)
        deadline = time.time() + self.stop_timeout
        while running:
            for rec in list(running):
                if _has_exited(rec.process) or time.time() >= deadline:
                    self._finish(rec)
                    rec.state = JobState.CANCELLED
                    self._event("stop", rec, "cancelled")
                    running.remove(rec)
            if running:
                time.sleep(min(self.poll_interval, 0.1))
        for rec in self._in_state(JobState.PENDING):
            rec.state = JobState.CANCELLED

    # -- helpers --------------------------------------------------------------

    def _finish(self, rec: JobRecord) -> int:
        """Kill whatever is left of the job's process group, then reap the leader.

        Leftover children (e.g. data-loader workers) would keep holding the
        job's GPU memory and cores, so a job ends when its leader does. The
        leader is reaped last so its pid, which is also the group id, cannot be
        reused before the group is killed.
        """
        self._signal_group(rec, signal.SIGKILL)
        rec.returncode = rec.process.wait()
        rec.ended_at = time.time()
        return rec.returncode

    def _signal_group(self, rec: JobRecord, sig: int) -> None:
        if rec.process is None:
            return
        try:
            if hasattr(os, "killpg"):
                os.killpg(rec.process.pid, sig)
            elif rec.process.poll() is None:
                rec.process.kill()
        except (ProcessLookupError, PermissionError):
            pass

    def _in_state(self, state: JobState) -> List[JobRecord]:
        return [r for r in self.records.values() if r.state is state]

    def _event(self, kind: str, rec: JobRecord, detail: str) -> None:
        self._say(f"{kind:<6} {rec.spec.name:<{self._name_width}}  {detail}", stamp=True)

    def _say(self, text: str, stamp: bool = False) -> None:
        if stamp:
            text = f"[{time.strftime('%H:%M:%S')}] {text}"
        print(text, file=self.out, flush=True)


def _has_exited(proc: subprocess.Popen) -> bool:
    """Whether ``proc`` has exited, without reaping it where the OS allows."""
    if proc.returncode is not None:
        return True
    if hasattr(os, "waitid"):
        try:
            return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        except ChildProcessError:
            return proc.poll() is not None
    return proc.poll() is not None
