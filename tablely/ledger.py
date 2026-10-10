"""Shared, machine-wide record of who is running what.

Every ``tablely run`` registers here, so agents (AI agents or people) running
Tablely at the same time on one machine

* plan GPUs and cores together: no GPU is booked twice, and priorities are
  compared across agents, not just within one job file;
* can see what everyone is working on (``tablely status``) and what happened
  (``tablely history``) without anyone having to write it down.

The ledger is a directory (``$TABLELY_HOME``, default ``~/.tablely``) holding
``state.json`` — the live board — and ``history.jsonl`` — an append-only event
log — both guarded by an exclusive ``flock`` on ``lock``. Runs whose process
has died are cleaned up by whoever looks next; jobs they left running keep
their resources until they exit.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX
    fcntl = None

from .planner import Allocation, Policy
from .resources import Inventory
from .spec import JobSpec

STATE_VERSION = 1
PENDING = "pending"
RUNNING = "running"


def default_home() -> Path:
    return Path(os.environ.get("TABLELY_HOME") or Path.home() / ".tablely")


def default_agent() -> str:
    """Who is acting: ``$TABLELY_AGENT``, else the login name."""
    name = os.environ.get("TABLELY_AGENT")
    if name:
        return name
    try:
        return getpass.getuser()
    except Exception:  # no passwd entry, e.g. in some containers
        return "agent"


def process_identity(pid: int) -> Optional[str]:
    """Kernel start time of ``pid``: tells a live process from a recycled pid (Linux)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return fields[19] if len(fields) > 19 else None  # field 22 of proc(5), counted after "comm"


def process_alive(pid: Optional[int], identity: Optional[str] = None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists, owned by someone else
    try:
        with open(f"/proc/{pid}/stat") as f:
            if f.read().rsplit(")", 1)[1].split()[0] == "Z":
                return False
    except (OSError, IndexError):
        pass
    if identity is not None:
        current = process_identity(pid)
        if current is not None and current != identity:
            return False
    return True


_RATIO = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def parse_fraction(text: Optional[str]) -> Optional[float]:
    """Read how far along a progress line is: ``"epoch 3/10"`` -> 0.3, ``"45%"`` -> 0.45."""
    if not text:
        return None
    ratio = _RATIO.search(text)
    if ratio and float(ratio.group(2)) > 0:
        return min(max(float(ratio.group(1)) / float(ratio.group(2)), 0.0), 1.0)
    percent = _PERCENT.search(text)
    if percent:
        return min(max(float(percent.group(1)) / 100, 0.0), 1.0)
    return None


def allocation_to_json(alloc: Optional[Allocation]) -> Optional[Dict[str, Any]]:
    if alloc is None:
        return None
    data = {"device": alloc.device, "gpus": list(alloc.gpus), "cpus": list(alloc.cpus)}
    if alloc.gpu_share is not None:
        data["gpu_share"] = alloc.gpu_share
    return data


def allocation_from_json(data: Optional[Mapping[str, Any]]) -> Optional[Allocation]:
    if not data:
        return None
    return Allocation(device=data["device"], gpus=tuple(data["gpus"]), cpus=tuple(data["cpus"]),
                      gpu_share=data.get("gpu_share"))


def pool_inventory(pool: Mapping[str, Any]) -> Inventory:
    """The shared pool recorded on the board, as an :class:`Inventory`."""
    return Inventory(cpus=tuple(pool["cpus"]), gpus=tuple(pool["gpus"]),
                     gpu_memory=tuple(pool.get("gpu_memory") or ()), ram=pool.get("ram"))


def _empty_state() -> Dict[str, Any]:
    return {"version": STATE_VERSION, "seq": 0, "pool": None, "runs": {}, "jobs": {}, "notes": {}}


class Board:
    """The live state, loaded under the lock. Mutations are saved on release."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data
        self.events: List[Dict[str, Any]] = []

    # -- reading --------------------------------------------------------------

    @property
    def runs(self) -> Dict[str, Dict[str, Any]]:
        return self.data["runs"]

    @property
    def jobs(self) -> Dict[str, Dict[str, Any]]:
        return self.data["jobs"]

    def job(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data["jobs"].get(key)

    def planning_view(self) -> Tuple[Dict[str, JobSpec], Dict[str, Allocation], List[str]]:
        """Every agent's jobs as planner input: specs (submission order), running, pending."""
        specs: Dict[str, JobSpec] = {}
        running: Dict[str, Allocation] = {}
        pending: List[str] = []
        for key, job in sorted(self.jobs.items(), key=lambda kv: (kv[1]["seq"], kv[0])):
            specs[key] = JobSpec(
                name=key,
                command="-",
                priority=job["priority"],
                device=job["device"],
                gpus=job["gpus"],
                cpus=job["cpus"],
                max_cpus=job["max_cpus"],
                max_gpus=job.get("max_gpus"),
                switchable=job.get("switchable", False),
                gpu_share=job.get("gpu_share"),
                gpu_memory=job.get("gpu_memory"),
                ram=job.get("ram"),
            )
            if job["state"] == RUNNING:
                running[key] = allocation_from_json(job["allocation"])
            else:
                pending.append(key)
        return specs, running, pending

    # -- writing --------------------------------------------------------------

    def next_seq(self) -> int:
        self.data["seq"] += 1
        return self.data["seq"]

    def add_run(self, run_id: str, info: Dict[str, Any]) -> None:
        self.runs[run_id] = info

    def remove_run(self, run_id: str) -> None:
        self.runs.pop(run_id, None)
        for key in [k for k, job in self.jobs.items() if job["run"] == run_id and not job.get("orphan")]:
            del self.jobs[key]
        self._reset_pool_if_idle()

    def put_job(self, key: str, fields: Dict[str, Any]) -> None:
        self.jobs[key] = fields

    def drop_job(self, key: str) -> None:
        self.jobs.pop(key, None)

    def set_note(self, agent: str, text: str) -> None:
        self.data["notes"][agent] = {"text": text, "at": time.time()}

    def log(self, event: Dict[str, Any]) -> None:
        self.events.append(event)

    def claim_pool(self, inventory: Inventory, policy: Policy, run_id: str) -> Tuple[Inventory, Policy, bool]:
        """Agree on one resource pool per machine.

        The first run sets the pool; while anything is registered, later runs
        use it even if their own settings differ, so everyone plans over the
        same cores and GPUs. Returns ``(inventory, policy, shared)``.
        """
        pool = self.data.get("pool")
        if pool and (self.runs or self.jobs):
            return pool_inventory(pool), Policy(backfill=pool["backfill"]), True
        self.data["pool"] = {
            "cpus": list(inventory.cpus),
            "gpus": list(inventory.gpus),
            "gpu_memory": list(inventory.gpu_memory),
            "ram": inventory.ram,
            "backfill": policy.backfill,
            "set_by": run_id,
            "set_at": time.time(),
        }
        return inventory, policy, False

    def prune(self) -> None:
        """Forget runs whose process died; keep their still-running jobs as orphans."""
        now = time.time()
        for run_id, run in list(self.runs.items()):
            if process_alive(run.get("pid"), run.get("pid_identity")):
                continue
            for key, job in list(self.jobs.items()):
                if job["run"] != run_id:
                    continue
                if job["state"] == RUNNING and process_alive(job.get("pid"), job.get("pid_identity")):
                    job["orphan"] = True  # it still holds its GPUs and cores
                else:
                    del self.jobs[key]
            del self.runs[run_id]
            self.log(make_event(now, "run-lost", run.get("agent"), run_id, None, run.get("task"),
                                "tablely process is gone; its queued jobs were dropped"))
        for key, job in list(self.jobs.items()):
            if job["run"] in self.runs:
                continue
            if not job.get("orphan") or not process_alive(job.get("pid"), job.get("pid_identity")):
                del self.jobs[key]
                if job.get("orphan"):
                    self.log(make_event(now, "orphan-exit", job.get("agent"), job["run"], job["name"],
                                        job.get("task"), "orphaned job finished; resources released"))
        self._reset_pool_if_idle()

    def _reset_pool_if_idle(self) -> None:
        if not self.runs and not self.jobs:
            self.data["pool"] = None


def make_event(
    t: float,
    kind: str,
    agent: Optional[str],
    run: Optional[str],
    job: Optional[str],
    task: Optional[str],
    detail: str,
    **extra: Any,
) -> Dict[str, Any]:
    event = {
        "t": round(t, 3),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)),
        "event": kind,
        "agent": agent,
        "run": run,
        "job": job,
        "task": task,
        "detail": detail,
    }
    event.update({k: v for k, v in extra.items() if v is not None})
    return event


class Ledger:
    """File-backed ledger shared by every Tablely process of this user on this machine."""

    shared = True

    def __init__(self, home: Union[str, Path, None] = None) -> None:
        self.home = (Path(home) if home is not None else default_home()).expanduser().absolute()
        self.state_path = self.home / "state.json"
        self.history_path = self.home / "history.jsonl"

    @contextlib.contextmanager
    def locked(self, save: bool = True) -> Iterator[Board]:
        """Hold the lock, yield the board, then save it and append its events."""
        self.home.mkdir(parents=True, exist_ok=True)
        with open(self.home / "lock", "a+") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                board = Board(self._load())
                try:
                    yield board
                finally:
                    if save:
                        self._save(board.data)
                        self._append(board.events)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def snapshot(self) -> Dict[str, Any]:
        """The current board with dead runs pruned, without changing anything on disk."""
        if not self.state_path.exists():
            return _empty_state()
        with self.locked(save=False) as board:
            board.prune()
            return board.data

    def history(
        self, agent: Optional[str] = None, job: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        try:
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        events = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except ValueError:  # a line cut short by a crash
                continue
        return _filter(events, agent, job, limit)

    def _load(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return _empty_state()
        except ValueError:
            self.state_path.replace(self.state_path.with_name(f"state.corrupt-{int(time.time())}.json"))
            return _empty_state()
        if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
            return _empty_state()
        return data

    def _save(self, data: Dict[str, Any]) -> None:
        tmp = self.state_path.with_name(f".state.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _append(self, events: List[Dict[str, Any]]) -> None:
        if not events:
            return
        with open(self.history_path, "a", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")


class MemoryLedger:
    """In-process stand-in for :class:`Ledger`: same interface, nothing shared."""

    shared = False
    home = None

    def __init__(self) -> None:
        self.data = _empty_state()
        self.events: List[Dict[str, Any]] = []

    @contextlib.contextmanager
    def locked(self, save: bool = True) -> Iterator[Board]:
        board = Board(self.data)
        try:
            yield board
        finally:
            self.events.extend(board.events)

    def snapshot(self) -> Dict[str, Any]:
        return self.data

    def history(
        self, agent: Optional[str] = None, job: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return _filter(self.events, agent, job, limit)


def _filter(events: List[Dict[str, Any]], agent: Optional[str], job: Optional[str], limit: Optional[int]):
    if agent:
        events = [e for e in events if e.get("agent") == agent]
    if job:
        events = [e for e in events if e.get("job") == job]
    if limit is not None:
        events = events[-limit:] if limit > 0 else []
    return events


def hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:  # pragma: no cover
        return "?"
