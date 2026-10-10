import sys
from pathlib import Path

from tablely.cli import main

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "jobs.toml"


def test_plan_previews_a_bigger_machine(capsys):
    assert main(["plan", str(EXAMPLE), "--cpus", "16", "--gpus", "2"]) == 0
    out = capsys.readouterr().out
    assert "15 CPU core(s)" in out  # one core reserved by the job file
    for name in ("llm-finetune", "vision-cnn", "xgboost-sweep", "tabular-mlp"):
        assert name in out
    assert "GPU 0" in out and "GPU 1" in out


def test_plan_reports_impossible_jobs(capsys):
    assert main(["plan", str(EXAMPLE), "--cpus", "16", "--gpus", "0"]) == 2
    assert "needs 1 GPU(s)" in capsys.readouterr().err


def test_bad_job_file_is_a_clean_error(tmp_path, capsys):
    bad = tmp_path / "jobs.toml"
    bad.write_text('[[jobs]]\nname = "a"\n')
    assert main(["plan", str(bad)]) == 2
    assert "missing 'command'" in capsys.readouterr().err


def test_run(tmp_path, capsys):
    jobfile = tmp_path / "jobs.toml"
    jobfile.write_text(
        f"""
[[jobs]]
name = "hello"
command = [{sys.executable!r}, "-c", "print('hi from {{name}} on {{device}}')"]
device = "cpu"
"""
    )
    assert main(["run", str(jobfile)]) == 0
    assert "hello" in capsys.readouterr().out
    assert (tmp_path / "tablely-logs" / "hello.log").read_text().strip() == "hi from hello on cpu"


def test_resources(capsys):
    assert main(["resources", "--gpus", "0"]) == 0
    assert "GPU(s) [none]" in capsys.readouterr().out


def test_plan_packs_small_jobs_onto_one_gpu(tmp_path, capsys):
    jobfile = tmp_path / "jobs.toml"
    jobfile.write_text("""
[[jobs]]
name = "big"
command = "x"
priority = 3

[[jobs]]
name = "half"
command = "x"
gpu_share = 0.5

[[jobs]]
name = "small"
command = "x"
gpu_memory = "6GiB"
""")
    assert main(["plan", str(jobfile), "--cpus", "8", "--gpus", "2", "--gpu-memory", "24GiB"]) == 0
    out = capsys.readouterr().out
    assert "2 GPU(s) [0 (24GiB), 1 (24GiB)]" in out
    rows = {line.split()[0]: line for line in out.splitlines() if line.startswith("  ")}
    assert "gpu x1" in rows["big"] and "GPU 0 " in rows["big"]
    assert "gpu x0.5" in rows["half"] and "GPU 1 (share 0.5)" in rows["half"]
    assert "gpu 6GiB" in rows["small"] and "GPU 1 (share 0.25)" in rows["small"]
    assert "gpu use: 0 100% · 1 75%" in out
