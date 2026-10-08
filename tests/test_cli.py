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
