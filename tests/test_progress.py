import io
import json
import os
import sys
import textwrap
import time

from tablely.cli import main
from tablely.ledger import Ledger, parse_fraction
from tablely.progress import ai_argv, collect_runs, resume_prompt, save_prompt
from tablely.resources import Inventory, detect_cpus
from tablely.runner import Runner
from tablely.spec import JobSpec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INV = Inventory(cpus=tuple(detect_cpus()), gpus=())


def py(code):
    return [sys.executable, "-c", textwrap.dedent(code)]


def reporting(text=None, **numbers):
    return py(f"""
        import sys
        sys.path.insert(0, {REPO!r})
        from tablely import client
        client.progress({text!r}, **{numbers!r})
    """)


def test_parse_fraction():
    assert parse_fraction("epoch 3/10, loss 0.41") == 0.3
    assert parse_fraction("72.5% of shards") == 0.725
    assert parse_fraction("12/10") == 1.0
    assert parse_fraction("loss 0.41") is None and parse_fraction("0/0") is None


def run_batch(tmp_path, ledger):
    jobs = [
        JobSpec(name="a", command=reporting("epoch 10/10"), device="cpu", task="sweep a"),
        JobSpec(name="b", command=reporting(done=3, total=4), device="cpu"),
        JobSpec(name="c", command=py("raise SystemExit(1)"), device="cpu"),
    ]
    runner = Runner(INV, jobs, log_dir=tmp_path / "logs", out=io.StringIO(), poll_interval=0.02,
                    ledger=ledger, agent="claude-a", task="lr sweep")
    runner.run()
    return runner


def test_completion_rate_of_a_run(tmp_path, isolated_tablely_home):
    ledger = Ledger(isolated_tablely_home)
    run_batch(tmp_path, ledger)
    (run,) = collect_runs(ledger.snapshot(), ledger.history())
    assert not run.live and run.agent == "claude-a" and run.task == "lr sweep"
    states = {j.name: j.state for j in run.jobs}
    assert states == {"a": "ok", "b": "ok", "c": "failed"}
    assert abs(run.achievement - 2 / 3) < 1e-9
    b = next(j for j in run.jobs if j.name == "b")
    assert b.progress == "3/4"  # the last report survives the job
    assert next(j for j in run.jobs if j.name == "a").task == "sweep a"


def test_running_jobs_count_by_reported_fraction():
    data = {
        "runs": {"r1": {"agent": "x", "task": "t", "started_at": 1.0, "jobs": ["j1", "j2"]}},
        "jobs": {
            "r1.j1": {"run": "r1", "agent": "x", "name": "j1", "state": "running", "progress": "epoch 1/4",
                      "progress_frac": 0.25, "allocation": {"device": "gpu", "gpus": ["0"], "cpus": [0]}},
            "r1.j2": {"run": "r1", "agent": "x", "name": "j2", "state": "pending", "wait_reason": "needs 1 GPU(s)"},
        },
    }
    (run,) = collect_runs(data, [])
    assert run.live and run.achievement == 0.125
    prompt = resume_prompt([run], agent="x")
    assert "Overall: 12% done" in prompt
    assert "- [running 25%] j1: on GPU 0; progress: epoch 1/4" in prompt
    assert "- [waiting] j2: needs 1 GPU(s)" in prompt


def test_status_shows_progress_bars(tmp_path, capsys, isolated_tablely_home):
    ledger = Ledger(isolated_tablely_home)
    run_batch(tmp_path, ledger)  # finished: 2 ok, 1 failed
    with ledger.locked() as board:
        board.add_run("r1", {"agent": "claude-z", "task": "학습률 탐색", "pid": os.getpid(), "started_at": time.time(),
                             "jobs": ["j1", "j2"]})
        board.put_job("r1.j1", {
            "run": "r1", "agent": "claude-z", "name": "j1", "task": "학습률 탐색", "priority": 1, "device": "cpu",
            "gpus": 0, "cpus": 1, "max_cpus": None, "state": "running", "seq": board.next_seq(),
            "allocation": {"device": "cpu", "gpus": [], "cpus": [0]}, "pid": os.getpid(), "started_at": 1.0,
            "progress": "epoch 2/5", "progress_frac": 0.4, "progress_at": 1.0, "orphan": False,
        })
        board.put_job("r1.j2", {
            "run": "r1", "agent": "claude-z", "name": "j2", "task": None, "priority": 3, "device": "gpu",
            "gpus": 1, "cpus": 1, "max_cpus": None, "state": "pending", "seq": board.next_seq(),
            "wait_reason": "needs 1 GPU(s), 0 free", "orphan": False,
        })
    assert main(["status", "--repo", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "claude-z · 학습률 탐색" in out
    assert "████░░░░░░░░░░░░░░░░  20%" in out  # run: (0.4 + 0) / 2 of a 20-cell bar
    assert "████░░░░░░  40%  running  CPU · cores 0 (1) · prio 1" in out and "epoch 2/5" in out
    assert "░░░░░░░░░░    -  waiting  prio 3" in out and "needs 1 GPU(s), 0 free" in out
    assert "recently finished" in out and "claude-a · lr sweep" in out and "2 ok · 1 failed" in out
    assert "resume any of these with: tablely resume" in out


def test_summary_files_and_ai_arguments(tmp_path):
    path = save_prompt("hello", tmp_path / "resume", "run x/y")
    assert path.read_text() == "hello" and path.name.endswith("-run-x-y.md")
    assert ai_argv("claude", "hello", path) == ["claude", "hello"]
    assert ai_argv("my-ai --file {prompt_file} -q", "hello", path) == ["my-ai", "--file", str(path), "-q"]
