"""Cooperative CPU <-> GPU switching: checkpoint, exit 75, restart elsewhere."""

import io
import json
import os
import sys
import textwrap
import time

from tablely.ledger import Ledger, process_identity
from tablely.resources import Inventory, detect_cpus
from tablely.runner import SWITCH_EXIT, JobState, Runner
from tablely.spec import JobSpec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INV = Inventory(cpus=tuple(detect_cpus()), gpus=("0",))


def py(code):
    return [sys.executable, "-c", textwrap.dedent(code)]


def switching_job(marks, *, first="poll", later="finish", limit=8.0):
    """A switchable job that logs each run's device, then reacts as told.

    first/later: "poll" waits for a switch request and exits for it,
    "finish" exits 0, "want-cpu" asks to continue on CPU, "ignore" sleeps.
    """
    return py(f"""
        import json, os, sys, time
        sys.path.insert(0, {REPO!r})
        from tablely import client
        with open({str(marks)!r}, "a") as f:
            f.write(json.dumps({{"device": os.environ["TABLELY_DEVICE"], "restarts": client.restarts()}}) + "\\n")
        mode = {first!r} if client.restarts() == 0 else {later!r}
        if mode == "finish":
            sys.exit(0)
        if mode == "want-cpu":
            client.exit_for_switch("cpu")
        deadline = time.time() + {limit}
        while time.time() < deadline:
            if mode == "poll" and client.switch_requested():
                client.exit_for_switch()
            time.sleep(0.02)
    """)


def runs(marks):
    return [json.loads(line) for line in marks.read_text().splitlines()]


def make(tmp_path, jobs, ledger=None, **kw):
    return Runner(INV, jobs, log_dir=tmp_path / "logs", out=io.StringIO(), poll_interval=0.02,
                  stop_timeout=2, switch_grace=0, ledger=ledger, agent="me", **kw)


def tick_until(runner, condition, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        runner._tick()
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached; output:\n" + runner.out.getvalue())


def other_agent_job(ledger, *, state, priority, device="gpu", pid=None, gpus=("0",)):
    with ledger.locked() as board:
        board.add_run("r0", {"agent": "other", "task": None, "pid": os.getpid(),
                             "pid_identity": process_identity(os.getpid()), "started_at": time.time()})
        board.put_job("r0.big", {
            "run": "r0", "agent": "other", "name": "big", "task": "big training", "priority": priority,
            "device": device, "gpus": 1, "cpus": 1, "max_cpus": None, "max_gpus": None, "switchable": False,
            "command": "x", "git": None, "state": state, "seq": board.next_seq(), "submitted_at": time.time(),
            "wait_reason": None, "switch_requested": None, "restarts": 0, "orphan": False,
            "allocation": {"device": "gpu", "gpus": list(gpus), "cpus": [INV.cpus[0]]} if state == "running" else None,
            "pid": pid, "pid_identity": process_identity(pid) if pid else None, "started_at": time.time(),
            "log": None, "progress": None, "progress_at": None,
        })


def test_cpu_job_moves_to_a_gpu_when_one_frees_up(tmp_path):
    marks = tmp_path / "marks"
    jobs = [
        JobSpec(name="hog", command=py("import time; time.sleep(0.6)"), device="gpu", priority=9),
        JobSpec(name="flex", command=switching_job(marks), device="any", switchable=True),
    ]
    runner = make(tmp_path, jobs)
    assert runner.run() == 0, runner.out.getvalue()
    assert runs(marks) == [{"device": "cpu", "restarts": 0}, {"device": "cuda", "restarts": 1}]
    out = runner.out.getvalue()
    assert "asked to checkpoint and move to GPU" in out and "restarting on GPU" in out
    assert "(1 device switch)" in runner.summary()


def test_more_important_gpu_job_pushes_a_switchable_job_to_cpu(tmp_path):
    marks = tmp_path / "marks"
    ledger = Ledger(tmp_path / "home")
    runner = make(tmp_path, [JobSpec(name="flex", command=switching_job(marks), device="any", switchable=True)],
                  ledger=ledger)
    runner._register()
    tick_until(runner, lambda: runner.records["flex"].state is JobState.RUNNING)
    assert runner.records["flex"].allocation.on_gpu

    other_agent_job(ledger, state="pending", priority=9)  # wants the GPU flex holds
    tick_until(runner, lambda: runner.records["flex"].restarts == 1 and runner.records["flex"].state is not JobState.PENDING)
    assert not runner.records["flex"].allocation.on_gpu  # restarted on CPU; the GPU is left for "big"
    tick_until(runner, lambda: runner.records["flex"].state is JobState.SUCCEEDED)
    assert runs(marks) == [{"device": "cuda", "restarts": 0}, {"device": "cpu", "restarts": 1}]
    assert "higher-priority big (other) needs a GPU" in runner.out.getvalue()
    runner._unregister()


def test_job_can_ask_to_continue_on_cpu(tmp_path):
    marks = tmp_path / "marks"
    job = JobSpec(name="flex", command=switching_job(marks, first="want-cpu", later="poll", limit=0.4),
                  device="any", switchable=True)
    runner = make(tmp_path, [job])
    assert runner.run() == 0, runner.out.getvalue()
    # restarted on CPU although the GPU is free, and never pulled back to it
    assert runs(marks) == [{"device": "cuda", "restarts": 0}, {"device": "cpu", "restarts": 1}]
    assert "restarting on CPU (job asked)" in runner.out.getvalue()


def test_request_is_withdrawn_when_the_spare_gpu_is_needed_elsewhere(tmp_path):
    marks = tmp_path / "marks"
    ledger = Ledger(tmp_path / "home")
    holder_pid = os.getpid()
    other_agent_job(ledger, state="running", priority=5, pid=holder_pid)  # holds GPU 0
    runner = make(tmp_path, [JobSpec(name="flex", command=switching_job(marks, first="ignore"),
                                     device="any", switchable=True)], ledger=ledger)
    runner._register()
    tick_until(runner, lambda: runner.records["flex"].state is JobState.RUNNING)
    assert not runner.records["flex"].allocation.on_gpu

    with ledger.locked() as board:  # the GPU frees up...
        board.drop_job("r0.big")
    tick_until(runner, lambda: runner.records["flex"].switch_requested == "gpu")
    control = tmp_path / "logs" / "flex.control"
    assert json.loads(control.read_text())["switch_to"] == "gpu"

    other_agent_job(ledger, state="pending", priority=9)  # ...but a GPU-only job claims it first
    tick_until(runner, lambda: runner.records["flex"].switch_requested is None)
    assert not control.exists()
    assert "request to move to gpu withdrawn" in runner.out.getvalue()
    runner._stop_all()
    runner._unregister()


def test_switch_exit_from_a_non_switchable_job_is_a_failure(tmp_path):
    runner = make(tmp_path, [JobSpec(name="plain", command=py(f"raise SystemExit({SWITCH_EXIT})"), device="cpu")])
    assert runner.run() == 1
    assert runner.records["plain"].returncode == SWITCH_EXIT


def test_endless_switching_is_capped(tmp_path):
    marks = tmp_path / "marks"
    job = JobSpec(name="flaky", command=switching_job(marks, first="want-cpu", later="want-cpu"),
                  device="any", switchable=True)
    runner = make(tmp_path, [job], max_switches=2)
    assert runner.run() == 1
    assert len(runs(marks)) == 3
    assert "more than 2 times" in runner.summary()
