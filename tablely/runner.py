"""Runs jobs as subprocesses according to the planner's decisions."""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import json
import math
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, TextIO, Tuple, Union

from . import _fmt, affinity
from .ledger import (
    PENDING,
    RUNNING,
    Board,
    Ledger,
    MemoryLedger,
    allocation_to_json,
    default_agent,
    hostname,
    make_event,
    process_identity,
)
from .planner import EPS, GPU, Allocation, Plan, Policy, check_feasible, fit_gpus, plan
from .resources import Inventory, format_cpu_list
from .spec import Device, JobSpec, format_priority

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_PLACEHOLDER = re.compile(r"\{(name|device|gpus|num_gpus|cpus|cpu_list)\}")

SWITCH_EXIT = 75  # EX_TEMPFAIL: "checkpointed, restart me" (see client.exit_for_switch)


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
    restarts: int = 0  # device switches so far
    switch_requested: Optional[str] = None  # "gpu"/"cpu" asked of the running process
    force_cpu: bool = False  # the job asked to stay on CPU (e.g. after running out of GPU memory)


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


SHARE_ENV_VARS = ("TABLELY_GPU_SHARE", "TABLELY_GPU_MEMORY", "TABLELY_RAM")


def sharing_env(spec: JobSpec, alloc: Allocation, gpu_total: Optional[int], mps: bool) -> Dict[str, str]:
    """Variables that keep a job within its part of a shared GPU.

    The GPU itself cannot stop a process from taking all of its memory, so
    frameworks are told: JAX preallocates only its share, TensorFlow grows
    on demand instead of grabbing everything, and PyTorch scripts call
    ``client.limit_gpu_memory()``. With MPS the CUDA driver also enforces a
    compute (SM) share and a memory limit.
    """
    share = alloc.gpu_share
    if not alloc.on_gpu or share is None:
        return {}
    limit = spec.gpu_memory if spec.gpu_memory is not None else (int(share * gpu_total) if gpu_total else None)
    env = {
        "TABLELY_GPU_SHARE": f"{share:.6g}",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": f"{math.floor(share * 1000) / 1000:g}",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    }
    if limit:
        env["TABLELY_GPU_MEMORY"] = str(limit)
    if mps:
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(max(1, round(share * 100)))
        if limit:
            env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = f"0={max(1, limit // 1024 ** 2)}MB"
    return env


def build_env(
    base: Mapping[str, str],
    spec: JobSpec,
    alloc: Allocation,
    extra: Optional[Mapping[str, str]] = None,
    *,
    gpu_total: Optional[int] = None,
    mps: bool = False,
) -> Dict[str, str]:
    """Environment for a job: its own ``env`` plus the resources it was given.

    GPU visibility is always enforced. Thread-count variables follow the core
    count unless the job sets them itself; so do the shared-GPU limits (see
    :func:`sharing_env`; ``gpu_total`` is the memory of the job's GPU).
    ``extra`` adds Tablely's own bookkeeping variables (agent, run, ledger
    location).
    """
    env = dict(base)
    for var in SHARE_ENV_VARS:  # never inherit another job's GPU share or RAM
        env.pop(var, None)
    env.update(spec.env)
    values = job_values(spec, alloc)
    for var in THREAD_ENV_VARS:
        if var not in spec.env:
            env[var] = values["cpus"]
    for var, value in sharing_env(spec, alloc, gpu_total, mps).items():
        if var not in spec.env or var in SHARE_ENV_VARS:
            env[var] = value
    if spec.ram:
        env["TABLELY_RAM"] = str(spec.ram)  # what tablely.tables may pin for this job
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
    if extra:
        env.update(extra)
    return env


def git_info(cwd: Optional[str]) -> Optional[Dict[str, Any]]:
    """Commit, branch and dirty flag of the repository a job runs in, if any."""

    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            raise OSError(result.stderr)
        return result.stdout.strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        branch = git("branch", "--show-current") or None
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "branch": branch, "dirty": dirty}


def command_text(spec: JobSpec) -> str:
    return spec.command if isinstance(spec.command, str) else shlex.join(spec.command)


class Runner:
    """Runs a batch of jobs to completion.

    Every scheduling round goes through a ledger. With the shared
    :class:`~tablely.ledger.Ledger` (what ``tablely run`` uses), all Tablely
    processes on the machine plan GPUs and cores together and every start,
    finish and resize is recorded with the agent and task behind it. The
    default in-memory ledger keeps the run to itself.
    """

    def __init__(
        self,
        inventory: Inventory,
        jobs: Sequence[JobSpec],
        *,
        policy: Policy = Policy(),
        log_dir: Union[str, Path] = "tablely-logs",
        poll_interval: float = 0.5,
        stop_timeout: float = 10.0,
        switch_grace: float = 60.0,
        max_switches: int = 5,
        out: Optional[TextIO] = None,
        base_env: Optional[Mapping[str, str]] = None,
        ledger: Union[Ledger, MemoryLedger, None] = None,
        agent: Optional[str] = None,
        task: Optional[str] = None,
        mps: bool = False,
    ) -> None:
        errors, self.warnings = check_feasible(inventory, jobs)
        if errors:
            raise ValueError("\n".join(errors))
        self.inventory = inventory
        self.policy = policy
        self.log_dir = Path(log_dir)
        self.poll_interval = poll_interval
        self.stop_timeout = stop_timeout
        self.switch_grace = switch_grace  # a job runs at least this long before being asked to move
        self.max_switches = max_switches
        self.out = out if out is not None else sys.stdout
        self.base_env = dict(os.environ if base_env is None else base_env)
        self.ledger = ledger if ledger is not None else MemoryLedger()
        self.agent = agent or default_agent()
        self.task = task
        self.mps = mps  # an MPS daemon runs: also set CUDA_MPS_* limits for shared GPUs
        self.run_id = secrets.token_hex(4)
        self.specs: Dict[str, JobSpec] = {spec.name: spec for spec in jobs}
        self.records: Dict[str, JobRecord] = {spec.name: JobRecord(spec) for spec in jobs}
        self._keys = {name: f"{self.run_id}.{name}" for name in self.specs}
        self._names = {key: name for name, key in self._keys.items()}
        self._run_info: Dict[str, Any] = {}
        self._shadow: Dict[str, Dict[str, Any]] = {}  # our last view of each job entry
        self._outbox: List[Dict[str, Any]] = []
        self._git: Dict[Optional[str], Optional[Dict[str, Any]]] = {}
        self._name_width = max((len(name) for name in self.specs), default=4)

    # -- public ---------------------------------------------------------------

    def run(self) -> int:
        """Run every job. Returns 0 if all succeeded, 1 if any failed, 130 if interrupted."""
        for spec in self.specs.values():
            if spec.cwd not in self._git:
                self._git[spec.cwd] = git_info(spec.cwd)
        self._register()
        self._say(f"agent: {self.agent}, run {self.run_id}" + (f", task: {self.task}" if self.task else ""))
        self._say(f"resources: {self.inventory.describe()}")
        self._say(f"policy: {'backfill' if self.policy.backfill else 'strict priority'}, logs in {self.log_dir}")
        if self.ledger.shared:
            self._say(f"shared with other agents via {self.ledger.home} (see: tablely status)")
        for warning in self.warnings:
            self._say(f"warning: {warning}")
        interrupted = False
        try:
            while True:
                self._tick()
                if not self._in_state(JobState.RUNNING) and not self._in_state(JobState.PENDING):
                    break
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            interrupted = True
            self._say("interrupted: stopping running jobs")
        finally:
            self._stop_all()
            self._unregister()
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
            if rec.restarts:
                result += f" ({rec.restarts} device switch{'es' if rec.restarts > 1 else ''})"
            took = (
                _fmt.duration(rec.ended_at - rec.started_at)
                if rec.started_at is not None and rec.ended_at is not None
                else "-"
            )
            rows.append(
                [rec.spec.name, format_priority(rec.spec.priority), _fmt.placement(rec.allocation), result, took]
            )
        return "summary:\n" + _fmt.table(["JOB", "PRIO", "RAN ON", "RESULT", "TIME"], rows)

    # -- ledger bookkeeping ---------------------------------------------------

    def _register(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        with self.ledger.locked() as board:
            board.prune()
            inventory, policy, shared = board.claim_pool(self.inventory, self.policy, self.run_id)
            if shared and (inventory != self.inventory or policy != self.policy):
                errors, warnings = check_feasible(inventory, list(self.specs.values()))
                if errors:
                    raise ValueError(
                        "other agents are running Tablely on this machine and share a different pool "
                        f"({inventory.describe()}):\n" + "\n".join(errors)
                    )
                mode = "backfill" if policy.backfill else "strict priority"
                self.warnings = warnings + [f"other agents already set this machine's pool; using it ({mode})"]
            self.inventory, self.policy = inventory, policy
            self._run_info = {
                "agent": self.agent,
                "task": self.task,
                "pid": os.getpid(),
                "pid_identity": process_identity(os.getpid()),
                "host": hostname(),
                "cwd": os.getcwd(),
                "started_at": now,
                "jobs": list(self.specs),
            }
            board.add_run(self.run_id, dict(self._run_info))
            for name in self.specs:
                self._put(board, name, self._new_entry(name, board.next_seq(), now))
            self._outbox.append(
                make_event(now, "run-start", self.agent, self.run_id, None, self.task,
                           f"{len(self.specs)} job(s): {', '.join(self.specs)}", jobs=list(self.specs),
                           job_tasks={n: s.task for n, s in self.specs.items() if s.task})
            )
            self._flush(board)

    def _unregister(self) -> None:
        with self.ledger.locked() as board:
            for key in self._keys.values():
                board.drop_job(key)
            board.remove_run(self.run_id)
            states = [r.state for r in self.records.values()]
            detail = ", ".join(
                f"{states.count(s)} {s.value}" for s in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)
                if states.count(s)
            )
            self._outbox.append(make_event(time.time(), "run-end", self.agent, self.run_id, None, self.task, detail))
            self._flush(board)

    def _new_entry(self, name: str, seq: int, now: float) -> Dict[str, Any]:
        spec = self.specs[name]
        return {
            "run": self.run_id,
            "agent": self.agent,
            "name": name,
            "task": spec.task or self.task,
            "priority": spec.priority,
            "device": spec.device.value,
            "gpus": spec.gpus,
            "cpus": spec.cpus,
            "max_cpus": spec.max_cpus,
            "max_gpus": spec.max_gpus,
            "gpu_share": spec.gpu_share,
            "gpu_memory": spec.gpu_memory,
            "ram": spec.ram,
            "tables": None,
            "switchable": spec.switchable,
            "switch_requested": None,
            "restarts": 0,
            "command": command_text(spec),
            "git": self._git.get(spec.cwd),
            "state": PENDING,
            "seq": seq,
            "submitted_at": now,
            "wait_reason": None,
            "allocation": None,
            "pid": None,
            "pid_identity": None,
            "started_at": None,
            "log": None,
            "progress": None,
            "progress_frac": None,
            "progress_at": None,
            "orphan": False,
        }

    def _put(self, board: Board, name: str, entry: Dict[str, Any]) -> None:
        board.put_job(self._keys[name], entry)
        self._shadow[name] = entry

    def _sync(self, board: Board) -> None:
        """Re-add our run and unfinished jobs if the board lost them (e.g. state file removed)."""
        if self.run_id not in board.runs:
            board.add_run(self.run_id, dict(self._run_info))
        for name, rec in self.records.items():
            key = self._keys[name]
            if rec.state in (JobState.PENDING, JobState.RUNNING):
                entry = board.job(key)
                if entry is None:
                    self._put(board, name, dict(self._shadow[name], seq=board.next_seq()))
                else:
                    self._shadow[name] = entry

    def _flush(self, board: Board) -> None:
        for event in self._outbox:
            board.log(event)
        self._outbox.clear()

    # -- scheduling -----------------------------------------------------------

    def _tick(self) -> None:
        with self.ledger.locked() as board:
            board.prune()
            self._sync(board)
            self._reap(board)
            self._schedule(board)
            self._flush(board)

    def _schedule(self, board: Board) -> None:
        while True:
            specs, running, pending = board.planning_view()
            decision = plan(self.inventory, specs, running, pending, self.policy)

            for key, alloc in decision.allocations.items():
                entry = board.job(key)
                if entry["state"] == RUNNING and list(alloc.cpus) != entry["allocation"]["cpus"]:
                    entry["allocation"]["cpus"] = list(alloc.cpus)
                    if key not in self._names:
                        # Another agent's job: move it now so a job we start here never
                        # shares cores with it; its own runner confirms on its next tick.
                        affinity.pin_group(entry["pid"], alloc.cpus)
            for key, reason in decision.waiting.items():
                board.job(key)["wait_reason"] = reason

            launch_failed = False
            for key in decision.started:
                name = self._names.get(key)
                if name is None:
                    board.job(key)["wait_reason"] = "starting"  # its own runner launches it
                elif not self._launch(self.records[name], decision.allocations[key], board):
                    launch_failed = True
            if not launch_failed:
                break  # otherwise re-plan: the failed job's resources are free again

        self._reconcile(board)
        self._steer_switches(board, specs, decision)
        for name, rec in self.records.items():
            if rec.state is not JobState.PENDING:
                continue
            reason = board.job(self._keys[name])["wait_reason"]
            if reason and reason != rec.wait_reason:
                rec.wait_reason = reason
                self._event("wait", rec, reason)

    # -- device switching -----------------------------------------------------

    def _control_files(self, name: str) -> Tuple[Path, Path]:
        """(requests from Tablely to the job, the job's own wish) for job ``name``."""
        return self.log_dir / f"{name}.control", self.log_dir / f"{name}.reply"

    def _steer_switches(self, board: Board, specs: Mapping[str, JobSpec], decision: Plan) -> None:
        """Ask switchable jobs to move between CPU and GPU, or take back a request.

        Every runner computes the same wishes from the shared board and only
        acts on its own jobs, so two agents never both claim one spare GPU.
        """
        wanted = _switch_wishes(board, specs, decision, self.inventory, time.time(),
                                self.switch_grace, self.max_switches)
        for name, rec in self.records.items():
            if rec.state is not JobState.RUNNING or not rec.spec.switchable:
                continue
            key = self._keys[name]
            target, reason = wanted.get(key, (None, None))
            if target == rec.switch_requested:
                continue
            control, _ = self._control_files(name)
            if target is None:
                with contextlib.suppress(FileNotFoundError):
                    control.unlink()
                self._event("switch", rec, f"request to move to {rec.switch_requested} withdrawn")
            else:
                tmp = control.with_name(control.name + ".tmp")
                tmp.write_text(json.dumps({"switch_to": target, "reason": reason, "at": time.time()}))
                os.replace(tmp, control)
                self._event("switch", rec, f"asked to checkpoint and move to {target.upper()}: {reason}")
            rec.switch_requested = target
            board.job(key)["switch_requested"] = target

    def _requeue(self, rec: JobRecord, board: Board) -> None:
        """The job checkpointed and exited to change devices: queue it again."""
        name = rec.spec.name
        _, reply = self._control_files(name)
        try:
            want = json.loads(reply.read_text()).get("want")
        except (OSError, ValueError, AttributeError):
            want = None
        if want == "cpu" and rec.spec.device is Device.ANY:
            rec.force_cpu = True
        moved_from = _fmt.placement(rec.allocation)
        asked = rec.switch_requested
        rec.restarts += 1
        rec.state = JobState.PENDING
        rec.process = None
        rec.allocation = None
        rec.switch_requested = None
        rec.wait_reason = None
        entry = board.job(self._keys[name])
        entry.update(
            state=PENDING,
            allocation=None,
            pid=None,
            pid_identity=None,
            switch_requested=None,
            restarts=rec.restarts,
            wait_reason="restarting on another device",
            device="cpu" if rec.force_cpu else rec.spec.device.value,
        )
        self._shadow[name] = entry
        target = "CPU (job asked)" if rec.force_cpu else (asked.upper() if asked else "next free device")
        self._event("requeue", rec, f"checkpointed on {moved_from}; restarting on {target} "
                                    f"(switch {rec.restarts}/{self.max_switches})")

    def _reconcile(self, board: Board) -> None:
        """Apply core changes the board holds for our running jobs (whoever planned them)."""
        for name, rec in self.records.items():
            if rec.state is not JobState.RUNNING:
                continue
            entry = board.job(self._keys[name])
            cpus = tuple(entry["allocation"]["cpus"])
            if cpus != rec.allocation.cpus:
                self._resize(rec, cpus)

    def _launch(self, rec: JobRecord, alloc: Allocation, board: Board) -> bool:
        spec = rec.spec
        key = self._keys[spec.name]
        rec.log_path = self.log_dir / f"{spec.name}.log"
        for path in self._control_files(spec.name):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
        control, reply = self._control_files(spec.name)
        extra = {
            "TABLELY_AGENT": self.agent,
            "TABLELY_RUN": self.run_id,
            "TABLELY_JOB_KEY": key,
            "TABLELY_SWITCHABLE": "1" if spec.switchable else "0",
            "TABLELY_RESTARTS": str(rec.restarts),
            "TABLELY_CONTROL": str(control.resolve()),
            "TABLELY_REPLY": str(reply.resolve()),
        }
        task = spec.task or self.task
        if task:
            extra["TABLELY_TASK"] = task
        if self.ledger.home is not None:
            extra["TABLELY_HOME"] = str(self.ledger.home)
        try:
            with open(rec.log_path, "ab" if rec.restarts else "wb") as log:  # keep earlier attempts
                rec.process = subprocess.Popen(
                    render_command(spec, alloc),
                    shell=spec.shell,
                    cwd=spec.cwd,
                    env=build_env(self.base_env, spec, alloc, extra, mps=self.mps,
                                  gpu_total=self.inventory.memory_of(alloc.gpus[0]) if alloc.gpus else None),
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
            board.drop_job(key)
            self._event("fail", rec, rec.error)
            return False
        rec.state = JobState.RUNNING
        rec.allocation = alloc
        rec.started_at = time.time()
        rec.wait_reason = None
        rec.switch_requested = None
        entry = board.job(key)
        entry.update(
            state=RUNNING,
            allocation=allocation_to_json(alloc),
            pid=rec.process.pid,
            pid_identity=process_identity(rec.process.pid),
            started_at=rec.started_at,
            wait_reason=None,
            switch_requested=None,
            log=str(rec.log_path.resolve()),
        )
        self._shadow[spec.name] = entry
        self._event(
            "start",
            rec,
            f"prio {format_priority(spec.priority)}  {_fmt.placement(alloc)}  cores {_fmt.cores(alloc)}",
            device=alloc.device,
            gpus=list(alloc.gpus),
            gpu_share=alloc.gpu_share,
            cpus=list(alloc.cpus),
            git=entry.get("git"),
        )
        return True

    def _resize(self, rec: JobRecord, cpus: Sequence[int]) -> None:
        old = rec.allocation
        rec.allocation = dataclasses.replace(old, cpus=tuple(cpus))
        affinity.pin_group(rec.process.pid, cpus)
        self._event("resize", rec, f"cores {_fmt.cores(old)} -> {_fmt.cores(rec.allocation)}", cpus=list(cpus))

    def _reap(self, board: Board) -> None:
        for name, rec in self.records.items():
            if rec.state is not JobState.RUNNING or not _has_exited(rec.process):
                continue
            code = self._finish(rec)
            if code == SWITCH_EXIT and rec.spec.switchable:
                if rec.restarts < self.max_switches:
                    self._requeue(rec, board)
                    continue
                rec.error = f"asked to switch devices more than {self.max_switches} times"
            rec.state = JobState.SUCCEEDED if code == 0 else JobState.FAILED
            self._shadow[name] = board.job(self._keys[name]) or self._shadow[name]
            board.drop_job(self._keys[name])
            took = _fmt.duration(rec.ended_at - rec.started_at)
            outcome = self._outcome(name, returncode=code, seconds=round(rec.ended_at - rec.started_at, 1))
            if code == 0:
                self._event("done", rec, f"ok in {took}", **outcome)
            else:
                self._event("fail", rec, f"exit {code} after {took}, see {rec.log_path}", **outcome)

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
                    self._event("stop", rec, "cancelled", **self._outcome(rec.spec.name))
                    running.remove(rec)
            if running:
                time.sleep(min(self.poll_interval, 0.1))
        for rec in self._in_state(JobState.PENDING):
            rec.state = JobState.CANCELLED

    # -- helpers --------------------------------------------------------------

    def _outcome(self, name: str, **extra: Any) -> Dict[str, Any]:
        """What the next agent needs about a finished job: last progress, code version, log."""
        entry = self._shadow.get(name) or {}
        return dict(extra, progress=entry.get("progress"), progress_frac=entry.get("progress_frac"),
                    git=entry.get("git"), log=entry.get("log"))

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

    def _event(self, kind: str, rec: JobRecord, detail: str, **extra: Any) -> None:
        """Print an event and queue it for the shared history."""
        self._say(f"{kind:<6} {rec.spec.name:<{self._name_width}}  {detail}", stamp=True)
        self._outbox.append(
            make_event(time.time(), kind, self.agent, self.run_id, rec.spec.name,
                       rec.spec.task or self.task, detail, **extra)
        )

    def _say(self, text: str, stamp: bool = False) -> None:
        if stamp:
            text = f"[{time.strftime('%H:%M:%S')}] {text}"
        print(text, file=self.out, flush=True)


def _switch_wishes(
    board: Board,
    specs: Mapping[str, JobSpec],
    decision: Plan,
    inventory: Inventory,
    now: float,
    grace: float,
    max_switches: int,
) -> Dict[str, Tuple[str, str]]:
    """Which running switchable jobs (any agent) should move, as ``key -> (target, reason)``.

    * up: a job that runs on CPU only because GPUs were busy moves to a GPU
      nobody waits for (or, for a shared-GPU job, to room left on one),
      highest priority first;
    * down: if a GPU-only job is waiting, lower-priority jobs that can run on
      CPU give up their GPUs, lowest priority first — only if that frees enough.
      Jobs holding part of a GPU are not asked: that would not free a GPU.

    Jobs already asked keep their place (so requests are stable), others must
    have run ``grace`` seconds; a job stops being asked after ``max_switches``.
    """
    wishes: Dict[str, Tuple[str, str]] = {}

    def movable(job: Dict[str, Any], on_gpu: bool, target: str) -> bool:
        alloc = job.get("allocation") or {}
        return (
            job["state"] == RUNNING
            and job.get("switchable")
            and job["device"] == "any"
            and not job.get("orphan")
            and (alloc.get("device") == GPU) == on_gpu
            and job.get("restarts", 0) < max_switches
            and (job.get("switch_requested") == target or now - (job.get("started_at") or now) >= grace)
        )

    order = lambda kv: (-kv[1]["priority"], kv[1]["seq"])  # noqa: E731
    gpu_waiting = any(specs[k].device is not Device.CPU for k in decision.waiting)
    load = dict(decision.gpu_load)
    spare = list(decision.spare_gpus)
    for key, job in sorted(board.jobs.items(), key=order):
        if not movable(job, on_gpu=False, target="gpu") or gpu_waiting:
            continue
        slot = fit_gpus(specs[key], inventory, load, spare)
        if slot is None:
            continue
        gpus, share = slot
        if share is None:
            reason = f"{len(spare)} GPU(s) free and nobody waiting for them"
        else:
            reason = f"room on GPU {gpus[0]} and nobody waiting for it"
        wishes[key] = ("gpu", reason)
        for gpu in gpus:
            load[gpu] += 1.0 if share is None else share
        spare = [gpu for gpu in spare if load[gpu] <= EPS]

    blocked = sorted(
        (k for k in decision.waiting if specs[k].device is Device.GPU),
        key=lambda k: (-specs[k].priority, board.jobs[k]["seq"]),
    )
    if blocked:
        top = blocked[0]
        taken = {g for alloc in decision.allocations.values() for g in alloc.gpus}
        need = specs[top].gpus - sum(1 for g in inventory.gpus if g not in taken)
        candidates = sorted(
            (
                (k, j) for k, j in board.jobs.items()
                if movable(j, on_gpu=True, target="cpu") and j["priority"] < specs[top].priority
                and not j["allocation"].get("gpu_share")
            ),
            key=lambda kv: (kv[1].get("switch_requested") != "cpu", kv[1]["priority"], -kv[1]["seq"]),
        )
        picked, freed = [], 0
        for key, job in candidates:
            if freed >= need:
                break
            picked.append(key)
            freed += len(job["allocation"]["gpus"])
        if need > 0 and freed >= need:
            for key in picked:
                wishes[key] = ("cpu", f"higher-priority {board.jobs[top]['name']} "
                                      f"({board.jobs[top]['agent']}) needs a GPU")
    return wishes


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
