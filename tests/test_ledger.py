import io
import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

from tablely.board_view import render_history, render_status
from tablely.cli import main
from tablely.ledger import Ledger, MemoryLedger, process_identity
from tablely.planner import Policy
from tablely.resources import Inventory, detect_cpus
from tablely.runner import JobState, Runner
from tablely.spec import JobSpec

CPUS = tuple(detect_cpus())
INV = Inventory(cpus=CPUS, gpus=("0",))


def py(code):
    return [sys.executable, "-c", textwrap.dedent(code)]


def runner(tmp_path, jobs, ledger, agent, inventory=INV, **kw):
    return Runner(
        inventory,
        jobs,
        log_dir=tmp_path / f"logs-{agent}",
        out=io.StringIO(),
        poll_interval=0.02,
        stop_timeout=2,
        ledger=ledger,
        agent=agent,
        **kw,
    )


def dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def seed(ledger, *, run_pid, job_pid=None, state="running", agent="ghost", gpus=("0",), cpus=(0,)):
    """Put another agent's run with one job on the board."""
    with ledger.locked() as board:
        board.claim_pool(INV, Policy(), "r0")
        board.add_run("r0", {"agent": agent, "task": "old work", "pid": run_pid,
                             "pid_identity": process_identity(run_pid), "started_at": time.time()})
        board.put_job("r0.train", {
            "run": "r0", "agent": agent, "name": "train", "task": "old work", "priority": 1.0,
            "device": "gpu", "gpus": 1, "cpus": 1, "max_cpus": None, "command": "x", "git": None,
            "state": state, "seq": board.next_seq(), "submitted_at": time.time(), "wait_reason": None,
            "allocation": {"device": "gpu", "gpus": list(gpus), "cpus": list(cpus)} if state == "running" else None,
            "pid": job_pid, "pid_identity": process_identity(job_pid) if job_pid else None,
            "started_at": time.time(), "log": None, "progress": None, "progress_at": None, "orphan": False,
        })


def test_another_agents_job_holds_its_gpu(tmp_path):
    ledger = Ledger(tmp_path / "home")
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.6)"])
    seed(ledger, run_pid=os.getpid(), job_pid=holder.pid)  # "ghost" is alive (shares our pid)

    def release_when_holder_exits():
        # The ghost runner is us; play its part by dropping the job once its process ends.
        holder.wait()
        with ledger.locked() as board:
            board.remove_run("r0")
            board.drop_job("r0.train")

    out = tmp_path / "started"
    me = runner(tmp_path, [JobSpec(name="mine", command=py(f"open({str(out)!r}, 'w').write('x')"), device="gpu")],
                ledger, "me")
    me._register()
    me._tick()
    assert me.records["mine"].state is JobState.PENDING  # GPU 0 belongs to the ghost
    assert ledger.snapshot()["jobs"][me._keys["mine"]]["wait_reason"] == "needs 1 GPU(s), 0 free"
    release_when_holder_exits()
    me._tick()
    assert me.records["mine"].state is JobState.RUNNING
    assert me.records["mine"].allocation.gpus == ("0",)
    me._stop_all()
    me._unregister()


def test_dead_runner_leaves_an_orphan_that_keeps_its_resources(tmp_path):
    ledger = Ledger(tmp_path / "home")
    orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.8)"])
    seed(ledger, run_pid=dead_pid(), job_pid=orphan.pid)
    me = runner(tmp_path, [JobSpec(name="mine", command=py("pass"), device="gpu")], ledger, "me")
    assert me.run() == 0
    orphan.wait()
    events = [e["event"] for e in ledger.history()]
    assert events.index("run-lost") < events.index("orphan-exit") < events.index("start")
    started = next(e for e in ledger.history() if e["event"] == "start")
    exited = next(e for e in ledger.history() if e["event"] == "orphan-exit")
    assert started["t"] >= exited["t"]
    assert ledger.snapshot()["jobs"] == {}


def test_dead_runner_queued_jobs_are_dropped(tmp_path):
    ledger = Ledger(tmp_path / "home")
    seed(ledger, run_pid=dead_pid(), state="pending")
    data = ledger.snapshot()
    assert data["runs"] == {} and data["jobs"] == {} and data["pool"] is None


def test_two_agents_in_separate_processes_never_share_a_gpu(tmp_path):
    home = tmp_path / "home"
    jobfile = tmp_path / "a.toml"
    marker = tmp_path / "a.json"
    body = (
        "import json, os, time; t0 = time.time(); time.sleep(0.8); "
        f"json.dump([t0, time.time(), os.environ['CUDA_VISIBLE_DEVICES']], open({str(marker)!r}, 'w'))"
    )
    jobfile.write_text(
        f'task = "agent a work"\n[[jobs]]\nname = "a-train"\ncommand = {json.dumps([sys.executable, "-c", body])}\n'
        'device = "gpu"\npriority = 1\n'
    )
    env = dict(os.environ, TABLELY_HOME=str(home), PYTHONPATH=os.getcwd())
    agent_a = subprocess.Popen(
        [sys.executable, "-m", "tablely", "run", str(jobfile), "--gpus", "1", "--agent", "agent-a"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    ledger = Ledger(home)
    deadline = time.time() + 10
    while time.time() < deadline:
        jobs = ledger.snapshot()["jobs"]
        if any(j["state"] == "running" for j in jobs.values()):
            break
        time.sleep(0.02)
    else:
        pytest.fail("agent a never started its job: " + agent_a.communicate()[0])

    status = render_status(ledger.snapshot())
    assert "agent-a" in status and "agent a work" in status

    b_marker = tmp_path / "b.json"
    me = runner(
        tmp_path,
        [JobSpec(name="b-train", device="gpu", priority=9, command=py(
            f"import json, os, time; json.dump([time.time(), os.environ['CUDA_VISIBLE_DEVICES']], open({str(b_marker)!r}, 'w'))"
        ))],
        ledger,
        "agent-b",
        inventory=Inventory(cpus=CPUS, gpus=("0",)),
    )
    assert me.run() == 0
    assert agent_a.wait(timeout=10) == 0, agent_a.stdout.read()
    a_start, a_end, a_gpu = json.loads(marker.read_text())
    b_start, b_gpu = json.loads(b_marker.read_text())
    assert a_gpu == b_gpu == "0"
    assert b_start >= a_end  # higher priority, but no preemption: waits for the GPU
    agents = {e["agent"] for e in ledger.history()}
    assert agents == {"agent-a", "agent-b"}


def test_later_agents_adopt_the_shared_pool(tmp_path):
    ledger = Ledger(tmp_path / "home")
    seed(ledger, run_pid=os.getpid(), state="pending")
    small = Inventory(cpus=CPUS[:1], gpus=())
    me = runner(tmp_path, [JobSpec(name="c", command=py("pass"), device="cpu")], ledger, "me", inventory=small)
    me._register()
    assert me.inventory == INV  # the pool the first agent set
    assert any("pool" in w for w in me.warnings)

    too_big = runner(tmp_path, [JobSpec(name="g", command="x", device="gpu", gpus=1)], ledger, "other",
                     inventory=Inventory(cpus=CPUS, gpus=("0", "1")))
    with ledger.locked() as board:
        board.data["pool"]["gpus"] = []
    with pytest.raises(ValueError, match="share a different pool"):
        too_big._register()


def test_runner_restores_its_entries_if_the_state_file_vanishes(tmp_path):
    ledger = Ledger(tmp_path / "home")
    me = runner(tmp_path, [JobSpec(name="w", command="x", device="gpu", gpus=1)], ledger, "me",
                inventory=Inventory(cpus=CPUS, gpus=("0",)))
    # hold the GPU so the job stays queued
    seed(ledger, run_pid=os.getpid(), job_pid=os.getpid())
    me._register()
    ledger.state_path.unlink()
    seed(ledger, run_pid=os.getpid(), job_pid=os.getpid())
    me._tick()
    data = ledger.snapshot()
    assert me.run_id in data["runs"] and me._keys["w"] in data["jobs"]


def test_progress_from_inside_a_job_shows_up_on_the_board(tmp_path):
    ledger = Ledger(tmp_path / "home")
    seen = tmp_path / "seen.json"
    code = f"""
        import json, os, sys, time
        sys.path.insert(0, {os.getcwd()!r})
        from tablely import client, ledger
        ok = client.progress("epoch 2/5, loss 0.3")
        job = ledger.Ledger(os.environ["TABLELY_HOME"]).snapshot()["jobs"][os.environ["TABLELY_JOB_KEY"]]
        json.dump([ok, job["progress"], client.agent(), client.task()], open({str(seen)!r}, "w"))
    """
    me = runner(tmp_path, [JobSpec(name="p", command=py(code), device="cpu")], ledger, "me", task="demo")
    assert me.run() == 0, (tmp_path / "logs-me" / "p.log").read_text()
    assert json.loads(seen.read_text()) == [True, "epoch 2/5, loss 0.3", "me", "demo"]


def test_history_records_who_did_what(tmp_path):
    ledger = MemoryLedger()
    jobs = [
        JobSpec(name="ok", command=py("pass"), device="cpu", task="first"),
        JobSpec(name="bad", command=py("raise SystemExit(2)"), device="cpu"),
    ]
    me = runner(tmp_path, jobs, ledger, "claude-x", task="batch")
    assert me.run() == 1
    events = ledger.history()
    kinds = [(e["event"], e["job"]) for e in events]
    assert kinds[0] == ("run-start", None) and kinds[-1] == ("run-end", None)
    assert ("done", "ok") in kinds and ("fail", "bad") in kinds
    assert {e["agent"] for e in events} == {"claude-x"}
    tasks = {e["job"]: e["task"] for e in events if e["job"]}
    assert tasks == {"ok": "first", "bad": "batch"}
    start = next(e for e in events if e["event"] == "start" and e["job"] == "ok")
    assert start["git"] is None or "commit" in start["git"]
    assert "claude-x" in render_history(events)


def test_cli_status_history_and_note(tmp_path, capsys, isolated_tablely_home):
    assert main(["status"]) == 0
    assert "nobody is running" in capsys.readouterr().out
    assert main(["note", "--agent", "claude-y", "reading", "the", "planner"]) == 0
    capsys.readouterr()
    assert main(["status"]) == 0
    assert "reading the planner" in capsys.readouterr().out
    assert main(["history", "--agent", "claude-y", "--json"]) == 0
    event = json.loads(capsys.readouterr().out.strip())
    assert event["event"] == "note" and event["detail"] == "reading the planner"


def test_cli_plan_accounts_for_jobs_already_running(tmp_path, capsys, isolated_tablely_home):
    seed(Ledger(isolated_tablely_home), run_pid=os.getpid(), job_pid=os.getpid())
    jobfile = tmp_path / "jobs.toml"
    jobfile.write_text('[[jobs]]\nname = "mine"\ncommand = "x"\ndevice = "gpu"\n')
    assert main(["plan", str(jobfile)]) == 0
    out = capsys.readouterr().out
    assert "planning around 1 job(s) already here from: ghost" in out
    assert "wait: needs 1 GPU(s), 0 free" in out
    # an explicit machine description is a what-if: other agents are ignored
    assert main(["plan", str(jobfile), "--gpus", "1"]) == 0
    assert "start" in capsys.readouterr().out


def test_brief_hands_over_results_progress_and_notes(tmp_path, capsys, isolated_tablely_home):
    ledger = Ledger(isolated_tablely_home)
    report = f"""
        import sys
        sys.path.insert(0, {os.getcwd()!r})
        from tablely import client
        client.progress("epoch 5/5, val acc 0.91")
    """
    jobs = [
        JobSpec(name="good", command=py(report), device="cpu", task="baseline"),
        JobSpec(name="bad", command=py("raise SystemExit(3)"), device="cpu"),
    ]
    me = runner(tmp_path, jobs, ledger, "claude-a", task="lr sweep")
    assert me.run() == 1
    assert main(["note", "--agent", "claude-a", "next: try lr 3e-4"]) == 0
    capsys.readouterr()

    done = next(e for e in ledger.history() if e["event"] == "done")
    assert done["progress"] == "epoch 5/5, val acc 0.91"  # survives the job

    assert main(["brief"]) == 0
    text = capsys.readouterr().out
    assert "nothing running or queued" in text
    assert "[claude-a] good (baseline): ok" in text and "last progress: epoch 5/5, val acc 0.91" in text
    assert "[claude-a] bad (lr sweep): exit 3" in text
    assert "next: try lr 3e-4" in text
    assert main(["brief", "--agent", "someone-else"]) == 0
    assert "nothing finished in this window" in capsys.readouterr().out


def test_agents_share_one_gpu_through_the_ledger(tmp_path):
    GiB = 1024 ** 3
    ledger = Ledger(tmp_path / "home")
    inventory = Inventory(cpus=CPUS, gpus=("0",), gpu_memory=(24 * GiB,))
    sleeper = py("import time; time.sleep(5)")
    a = runner(tmp_path, [JobSpec(name="a", command=sleeper, gpu_memory="12GiB")], ledger, "alice", inventory)
    b = runner(tmp_path, [JobSpec(name="b", command=sleeper, gpu_share=0.5, priority=2)], ledger, "bob",
               inventory=Inventory(cpus=CPUS, gpus=("0",)))  # bob adopts alice's pool, memory included
    c = runner(tmp_path, [JobSpec(name="c", command=sleeper, gpu_share=0.25)], ledger, "carol", inventory)
    try:
        for r in (a, b, c):
            r._register()
            r._tick()
        assert b.inventory.memory_of("0") == 24 * GiB
        assert a.records["a"].allocation.gpu_share == 0.5 and b.records["b"].allocation.gpu_share == 0.5
        assert a.records["a"].allocation.gpus == b.records["b"].allocation.gpus == ("0",)
        assert c.records["c"].state is JobState.PENDING
        data = ledger.snapshot()
        assert data["jobs"][f"{c.run_id}.c"]["wait_reason"] == "needs 0.25 of a GPU, at most 0 free on one"
        view = render_status(data, events=ledger.history())
        assert "GPU(s) [0 (24GiB)]" in view
        assert "gpu use  0 ██████████ 100% 2 jobs" in view
        assert "GPU 0 (share 0.5)" in view
    finally:
        for r in (a, b, c):
            r._stop_all()
            r._unregister()


def test_status_shows_reserved_ram(tmp_path):
    ledger = Ledger(tmp_path / "home")
    inventory = Inventory(cpus=CPUS, gpus=(), ram=64 * 1024 ** 3)
    r = runner(tmp_path, [JobSpec(name="recsys", command=py("import time; time.sleep(5)"), device="cpu",
                                  ram="16GiB")], ledger, "alice", inventory)
    try:
        r._register()
        r._tick()
        view = render_status(ledger.snapshot())
        assert "ram use  ██░░░░░░░░  25% 16GiB of 64GiB reserved by 1 job" in view
        assert "ram 16GiB" in view
    finally:
        r._stop_all()
        r._unregister()
