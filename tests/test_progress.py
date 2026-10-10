import io
import json
import os
import sys
import textwrap

from tablely.cli import main
from tablely.ledger import Ledger, parse_fraction
from tablely.progress import collect_runs, hand_to_ai, resume_prompt
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


def test_status_shows_completion(tmp_path, capsys, isolated_tablely_home):
    ledger = Ledger(isolated_tablely_home)
    with ledger.locked() as board:
        board.add_run("r1", {"agent": "claude-z", "task": "t", "pid": os.getpid(), "started_at": 1.0,
                             "jobs": ["j1"]})
        board.put_job("r1.j1", {
            "run": "r1", "agent": "claude-z", "name": "j1", "task": "t", "priority": 1, "device": "cpu",
            "gpus": 0, "cpus": 1, "max_cpus": None, "state": "running", "seq": board.next_seq(),
            "allocation": {"device": "cpu", "gpus": [], "cpus": [0]}, "pid": os.getpid(), "started_at": 1.0,
            "progress": "epoch 2/5", "progress_frac": 0.4, "progress_at": 1.0, "orphan": False,
        })
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "DONE" in out and "40%" in out


def test_resume_command_summarizes_and_can_launch_an_ai(tmp_path, capsys, isolated_tablely_home):
    ledger = Ledger(isolated_tablely_home)
    run_batch(tmp_path, ledger)
    assert main(["note", "--agent", "claude-a", "next: rerun c with a smaller lr"]) == 0
    capsys.readouterr()

    assert main(["resume", "--agent", "claude-a", "--print-only"]) == 0
    captured = capsys.readouterr()
    prompt = captured.out
    assert prompt.startswith("# Resume: lr sweep (agent claude-a)")
    assert "Overall: 67% done. 2 of 3 jobs finished OK, 1 failed" in prompt
    assert "- [failed] c" in prompt and "c.log" in prompt
    assert "next: rerun c with a smaller lr" in prompt
    saved = captured.err.split("saved to ")[1].split(")")[0]
    assert open(saved).read() == prompt.rstrip("\n")

    got = tmp_path / "ai-got.json"
    fake_ai = f"{sys.executable} -c \"import json,sys; json.dump(sys.argv[1:], open({str(got)!r}, 'w'))\""
    assert main(["resume", "--agent", "claude-a", "--launch", fake_ai]) == 0
    assert "started" in capsys.readouterr().err
    for _ in range(200):
        if got.exists():
            break
        __import__("time").sleep(0.02)
    (handed,) = json.loads(got.read_text())
    assert handed.startswith("# Resume: lr sweep")


def test_prompt_file_placeholder(tmp_path):
    out = tmp_path / "seen.txt"
    result = hand_to_ai("hello", tmp_path / "resume", "x y/z",
                        command=f"{sys.executable} -c \"import sys,shutil; shutil.copy(sys.argv[1], {str(out)!r})\" {{prompt_file}}")
    assert result["launched"] and result["prompt_file"].endswith("-x-y-z.md")
    for _ in range(200):
        if out.exists():
            break
        __import__("time").sleep(0.02)
    assert out.read_text() == "hello"
    assert hand_to_ai("hi", tmp_path / "r", "l", command="definitely-not-an-ai-xyz")["error"]
