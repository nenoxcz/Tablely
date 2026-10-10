import io
import json
import os
import signal
import sys
import textwrap
import time

import pytest

from tablely import affinity
from tablely.planner import Allocation, Policy
from tablely.resources import Inventory, detect_cpus, parse_cpu_list
from tablely.runner import JobState, Runner, build_env, render_command
from tablely.spec import JobSpec

CPUS = tuple(detect_cpus())


def py(code):
    """A command running ``code`` with the test interpreter."""
    return [sys.executable, "-c", textwrap.dedent(code)]


def report(path, extra="", after=""):
    """Job body that dumps what Tablely gave it to ``path``."""
    return py(
        f"""
        import json, os, time
        {extra}
        info = {{k: v for k, v in os.environ.items() if k.startswith(("TABLELY_", "CUDA_", "OMP_"))}}
        if hasattr(os, "sched_getaffinity"):
            info["affinity"] = sorted(os.sched_getaffinity(0))
        info["t"] = time.time()
        json.dump(info, open({str(path)!r}, "w"))
        {after}
        """
    )


def make_runner(tmp_path, jobs, gpus=("0",), cpus=CPUS, **kw):
    kw.setdefault("poll_interval", 0.02)
    kw.setdefault("stop_timeout", 2)
    return Runner(
        Inventory(cpus=tuple(cpus), gpus=tuple(gpus)),
        jobs,
        log_dir=tmp_path / "logs",
        out=io.StringIO(),
        **kw,
    )


def test_render_command_and_env():
    spec = JobSpec(name="t", command="train.py --device {device} --gpus {gpus} -j {cpus}", env={"OMP_NUM_THREADS": "1"})
    alloc = Allocation("gpu", ("2", "3"), (4, 5, 6))
    assert render_command(spec, alloc) == ["train.py", "--device", "cuda", "--gpus", "2,3", "-j", "3"]
    env = build_env({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, spec, alloc)
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["TABLELY_DEVICE"] == "cuda"
    assert env["TABLELY_CPU_LIST"] == "4-6"
    assert env["MKL_NUM_THREADS"] == "3"
    assert env["OMP_NUM_THREADS"] == "1"  # the job's own setting wins

    cpu_env = build_env({}, JobSpec(name="c", command="x", device="cpu"), Allocation("cpu", (), (0,)))
    assert cpu_env["CUDA_VISIBLE_DEVICES"] == ""  # GPUs hidden from CPU jobs
    assert cpu_env["TABLELY_DEVICE"] == "cpu"


def test_jobs_receive_their_allocation(tmp_path):
    jobs = [
        # Both stay alive briefly so neither is resized before the other reports.
        JobSpec(name="gpu-job", command=report(tmp_path / "gpu.json", after="time.sleep(0.5)"), priority=5, device="gpu"),
        JobSpec(name="flex", command=report(tmp_path / "flex.json", after="time.sleep(0.5)"), priority=1, device="any"),
    ]
    runner = make_runner(tmp_path, jobs)
    assert runner.run() == 0
    gpu = json.loads((tmp_path / "gpu.json").read_text())
    flex = json.loads((tmp_path / "flex.json").read_text())
    assert gpu["CUDA_VISIBLE_DEVICES"] == "0" and gpu["TABLELY_DEVICE"] == "cuda"
    assert flex["CUDA_VISIBLE_DEVICES"] == "" and flex["TABLELY_DEVICE"] == "cpu"
    if affinity.SUPPORTED:
        assert gpu["affinity"] == parse_cpu_list(gpu["TABLELY_CPU_LIST"])
        assert not set(gpu["affinity"]) & set(flex["affinity"])
        assert len(flex["affinity"]) == int(flex["OMP_NUM_THREADS"])
    assert all(r.state is JobState.SUCCEEDED for r in runner.records.values())


def test_lower_priority_gpu_job_waits_for_the_gpu(tmp_path):
    jobs = [
        JobSpec(name="second", command=report(tmp_path / "second.json"), priority=1, device="gpu"),
        JobSpec(name="first", command=report(tmp_path / "first.json", "time.sleep(0.4)"), priority=9, device="gpu"),
    ]
    runner = make_runner(tmp_path, jobs)
    assert runner.run() == 0
    first = json.loads((tmp_path / "first.json").read_text())
    second = json.loads((tmp_path / "second.json").read_text())
    assert second["t"] > first["t"]
    assert runner.records["second"].started_at >= runner.records["first"].ended_at
    assert "wait" in runner.out.getvalue()


@pytest.mark.skipif(not affinity.SUPPORTED or len(CPUS) < 2, reason="needs Linux CPU affinity and 2+ cores")
def test_running_job_grows_when_a_neighbour_finishes(tmp_path):
    out = tmp_path / "grow.json"
    total = len(CPUS)
    grower = py(
        f"""
        import json, os, time
        start = sorted(os.sched_getaffinity(0))
        deadline = time.time() + 10
        while len(os.sched_getaffinity(0)) < {total} and time.time() < deadline:
            time.sleep(0.02)
        json.dump({{"start": start, "end": sorted(os.sched_getaffinity(0))}}, open({str(out)!r}, "w"))
        """
    )
    jobs = [
        JobSpec(name="grower", command=grower, device="cpu"),
        JobSpec(name="short", command=py("import time; time.sleep(0.3)"), device="cpu"),
    ]
    runner = make_runner(tmp_path, jobs, gpus=())
    assert runner.run() == 0
    result = json.loads(out.read_text())
    assert len(result["start"]) < total
    assert result["end"] == list(CPUS)
    assert "resize" in runner.out.getvalue()


def test_failures_are_reported_and_do_not_stop_other_jobs(tmp_path):
    jobs = [
        JobSpec(name="crash", command=py("raise SystemExit(3)"), priority=3, device="cpu"),
        JobSpec(name="missing", command="definitely-not-a-real-command-xyz", priority=2, device="cpu"),
        JobSpec(name="fine", command=py("print('hello')"), priority=1, device="cpu"),
    ]
    runner = make_runner(tmp_path, jobs, gpus=())
    assert runner.run() == 1
    records = runner.records
    assert records["crash"].state is JobState.FAILED and records["crash"].returncode == 3
    assert records["missing"].state is JobState.FAILED and "launch failed" in records["missing"].error
    assert records["fine"].state is JobState.SUCCEEDED
    assert (tmp_path / "logs" / "fine.log").read_text().strip() == "hello"
    assert "summary" in runner.out.getvalue()


def test_stop_all_terminates_the_whole_process_group(tmp_path):
    pid_file = tmp_path / "child.pid"
    parent = py(
        f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open({str(pid_file)!r}, "w").write(str(child.pid))
        time.sleep(60)
        """
    )
    jobs = [
        JobSpec(name="long", command=parent, device="cpu"),
        JobSpec(name="queued", command=py("pass"), device="gpu"),
    ]
    runner = make_runner(tmp_path, jobs, gpus=("0",), cpus=CPUS[:1])
    runner._register()
    runner._tick()
    deadline = time.time() + 5
    while not pid_file.exists() and time.time() < deadline:
        time.sleep(0.02)
    child_pid = int(pid_file.read_text())
    runner._stop_all()
    assert runner.records["long"].state is JobState.CANCELLED
    assert runner.records["queued"].state is JobState.CANCELLED
    deadline = time.time() + 5
    while _alive(child_pid) and time.time() < deadline:
        time.sleep(0.02)
    assert not _alive(child_pid)


def test_leftover_children_are_killed_when_the_job_ends(tmp_path):
    pid_file = tmp_path / "child.pid"
    leader = py(
        f"""
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open({str(pid_file)!r}, "w").write(str(child.pid))
        """
    )
    runner = make_runner(tmp_path, [JobSpec(name="leaky", command=leader, device="cpu")], gpus=())
    assert runner.run() == 0
    child_pid = int(pid_file.read_text())
    deadline = time.time() + 5
    while _alive(child_pid) and time.time() < deadline:
        time.sleep(0.02)
    assert not _alive(child_pid)


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs setitimer")
def test_interrupt_stops_jobs_and_returns_130(tmp_path):
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    jobs = [JobSpec(name="long", command=py("import time; time.sleep(60)"), device="cpu")]
    runner = make_runner(tmp_path, jobs, gpus=())
    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        started = time.time()
        assert runner.run() == 130
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert time.time() - started < 5
    assert runner.records["long"].state is JobState.CANCELLED
    assert "interrupted" in runner.out.getvalue()


def test_backfill_policy_is_passed_through(tmp_path):
    jobs = [JobSpec(name="a", command=py("pass"), device="cpu")]
    runner = make_runner(tmp_path, jobs, gpus=(), policy=Policy(backfill=True))
    assert runner.run() == 0
    assert "backfill" in runner.out.getvalue()


def test_infeasible_jobs_are_rejected_up_front(tmp_path):
    with pytest.raises(ValueError, match="needs 2 GPU"):
        make_runner(tmp_path, [JobSpec(name="a", command="x", device="gpu", gpus=2)], gpus=("0",))


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); treat it as dead.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def test_shared_gpu_jobs_get_memory_limits_in_their_env():
    GiB = 1024 ** 3
    spec = JobSpec(name="s", command="x", gpu_memory="6GiB")
    alloc = Allocation("gpu", ("1",), (0,), gpu_share=0.25)
    env = build_env({"TF_FORCE_GPU_ALLOW_GROWTH": "false"}, spec, alloc, gpu_total=24 * GiB)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["TABLELY_GPU_SHARE"] == "0.25" and env["TABLELY_GPU_MEMORY"] == str(6 * GiB)
    assert env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.25"
    assert env["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in env

    third = Allocation("gpu", ("0",), (0,), gpu_share=1 / 3)
    spec = JobSpec(name="s", command="x", gpu_share=1 / 3, env={"XLA_PYTHON_CLIENT_MEM_FRACTION": ".2"})
    env = build_env({}, spec, third, gpu_total=24 * GiB, mps=True)
    assert env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == ".2"  # the job's own setting wins
    assert env["TABLELY_GPU_MEMORY"] == str(8 * GiB)
    assert env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "33"
    assert env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "0=8192MB"

    whole = build_env({"TABLELY_GPU_SHARE": "0.5"}, JobSpec(name="w", command="x"), Allocation("gpu", ("0",), (0,)))
    assert "TABLELY_GPU_SHARE" not in whole and "XLA_PYTHON_CLIENT_MEM_FRACTION" not in whole


def test_small_jobs_run_side_by_side_on_one_gpu(tmp_path):
    jobs = [
        JobSpec(name=n, command=report(tmp_path / f"{n}.json", after="time.sleep(0.6)"), gpu_share=0.5)
        for n in ("a", "b")
    ]
    jobs.append(JobSpec(name="whole", command=report(tmp_path / "whole.json"), priority=0.5))
    runner = make_runner(tmp_path, jobs)
    assert runner.run() == 0, runner.out.getvalue()
    a, b, whole = (json.loads((tmp_path / f"{n}.json").read_text()) for n in ("a", "b", "whole"))
    assert a["CUDA_VISIBLE_DEVICES"] == b["CUDA_VISIBLE_DEVICES"] == "0"
    assert a["TABLELY_GPU_SHARE"] == b["TABLELY_GPU_SHARE"] == "0.5"
    assert abs(a["t"] - b["t"]) < 0.5  # both ran at once
    assert whole["t"] > max(a["t"], b["t"]) + 0.3  # the whole-GPU job waited for both
    assert "GPU 0 (share 0.5)" in runner.summary()


def test_client_reads_its_gpu_share(monkeypatch):
    from tablely import client

    monkeypatch.setenv("TABLELY_DEVICE", "cuda")
    monkeypatch.setenv("TABLELY_GPU_SHARE", "0.25")
    monkeypatch.setenv("TABLELY_GPU_MEMORY", str(6 * 1024 ** 3))
    assert client.gpu_share() == 0.25 and client.gpu_memory() == 6 * 1024 ** 3
    monkeypatch.setenv("TABLELY_DEVICE", "cpu")  # fell back to CPU: no GPU share
    assert client.gpu_share() is None and client.gpu_memory() is None
    assert client.limit_gpu_memory() is None  # nothing to limit, torch not even imported
