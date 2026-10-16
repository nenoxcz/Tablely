import io
import json
import sys
import textwrap

import pytest

from tablely.ledger import Ledger
from tablely.resources import Inventory, detect_cpus
from tablely.runner import JobState, Runner
from tablely.spec import JobSpec

INV = Inventory(cpus=tuple(detect_cpus()), gpus=("0",))


def py(code):
    return [sys.executable, "-c", textwrap.dedent(code)]


def stamp(path, extra=""):
    """A job that writes when it started and ended (and its attempt) to ``path``."""
    return py(f"""
        import json, os, time
        start = time.time()
        {extra}
        with open({str(path)!r}, "a") as f:
            f.write(json.dumps({{"start": start, "end": time.time(), "attempt": os.environ["TABLELY_ATTEMPT"]}}) + "\\n")
    """)


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def make(tmp_path, jobs, **kw):
    kw.setdefault("ledger", None)
    return Runner(INV, jobs, log_dir=tmp_path / "logs", out=io.StringIO(), poll_interval=0.02, stop_timeout=1, **kw)


def test_jobs_start_after_what_they_depend_on(tmp_path):
    jobs = [
        JobSpec(name="eval", command=stamp(tmp_path / "eval"), device="cpu", after=["train-a", "train-b"]),
        JobSpec(name="train-a", command=stamp(tmp_path / "a", "time.sleep(0.3)"), after=["prep"]),
        JobSpec(name="train-b", command=stamp(tmp_path / "b", "time.sleep(0.1)"), device="cpu", after=["prep"]),
        JobSpec(name="prep", command=stamp(tmp_path / "prep", "time.sleep(0.2)"), device="cpu"),
    ]
    runner = make(tmp_path, jobs)
    assert runner.run() == 0, runner.out.getvalue()
    prep, a, b, ev = (lines(tmp_path / n)[0] for n in ("prep", "a", "b", "eval"))
    assert a["start"] >= prep["end"] and b["start"] >= prep["end"]
    assert abs(a["start"] - b["start"]) < 0.25  # both stages of the fan-out run side by side
    assert ev["start"] >= max(a["end"], b["end"])
    assert "ready  eval" in runner.out.getvalue()


def test_waiting_jobs_are_not_planned_until_ready(tmp_path):
    ledger = Ledger(tmp_path / "home")
    jobs = [
        JobSpec(name="prep", command=py("import time; time.sleep(5)"), device="cpu"),
        JobSpec(name="train", command="true", priority=9, after=["prep"]),
    ]
    runner = make(tmp_path, jobs, ledger=ledger)
    runner._register()
    try:
        runner._tick()
        entry = ledger.snapshot()["jobs"][f"{runner.run_id}.train"]
        assert entry["state"] == "blocked" and entry["wait_reason"] == "after prep"
        with ledger.locked() as board:
            specs, running, pending = board.planning_view()
        assert f"{runner.run_id}.train" not in pending  # holds no GPU while it cannot start anyway
    finally:
        runner._stop_all()
        runner._unregister()


def test_a_failed_step_skips_everything_after_it(tmp_path):
    jobs = [
        JobSpec(name="prep", command=py("raise SystemExit(3)"), device="cpu"),
        JobSpec(name="train", command="true", device="cpu", after=["prep"]),
        JobSpec(name="eval", command="true", device="cpu", after=["train"]),
        JobSpec(name="other", command="true", device="cpu"),
    ]
    runner = make(tmp_path, jobs)
    assert runner.run() == 1
    states = {n: r.state for n, r in runner.records.items()}
    assert states == {"prep": JobState.FAILED, "train": JobState.CANCELLED, "eval": JobState.CANCELLED,
                      "other": JobState.SUCCEEDED}
    summary = runner.summary()
    assert "skipped: prep failed" in summary and "skipped: train cancelled" in summary


def test_failed_jobs_are_retried(tmp_path):
    marks = tmp_path / "marks"
    flaky = py(f"""
        import os
        with open({str(marks)!r}, "a") as f:
            f.write(os.environ["TABLELY_ATTEMPT"] + "\\n")
        raise SystemExit(1 if os.environ["TABLELY_ATTEMPT"] == "1" else 0)
    """)
    runner = make(tmp_path, [JobSpec(name="flaky", command=flaky, device="cpu", retries=2)])
    assert runner.run() == 0, runner.out.getvalue()
    assert marks.read_text().split() == ["1", "2"]
    assert "trying again (1/2)" in runner.out.getvalue()
    assert "(2 attempts)" in runner.summary()

    runner = make(tmp_path, [JobSpec(name="broken", command=py("raise SystemExit(2)"), device="cpu", retries=1)])
    assert runner.run() == 1
    assert "exit 2 (2 attempts)" in runner.summary()


def test_timeout_stops_a_job_and_counts_as_a_failure(tmp_path):
    jobs = [
        JobSpec(name="slow", command=py("import time; time.sleep(30)"), device="cpu", timeout=1),
        JobSpec(name="stubborn", device="cpu", timeout="1s",
                command=py("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)")),
    ]
    runner = make(tmp_path, jobs)
    assert runner.run() == 1
    for rec in runner.records.values():
        assert rec.state is JobState.FAILED and rec.error == "timed out after 1s"
        assert rec.ended_at - rec.started_at < 5  # SIGKILL followed when SIGTERM was ignored
    assert "ran longer than 1s; stopping it" in runner.out.getvalue()


def test_pipeline_validation():
    inventory = Inventory(cpus=(0,), gpus=())
    with pytest.raises(ValueError, match="no job named 'nope'"):
        Runner(inventory, [JobSpec(name="a", command="x", device="cpu", after=["nope"])])
    with pytest.raises(ValueError, match="circle"):
        Runner(inventory, [JobSpec(name="a", command="x", device="cpu", after=["b"]),
                           JobSpec(name="b", command="x", device="cpu", after=["a"])])
    for bad in ({"after": ["a"]}, {"retries": -1}, {"timeout": "soon"}):
        with pytest.raises(ValueError):
            JobSpec(name="a", command="x", **bad)
