import json
from pathlib import Path

import pytest

from tablely.config import ConfigError, load_config, parse_config
from tablely.spec import Device

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "jobs.toml"


def test_example_job_file_loads():
    config = load_config(EXAMPLE)
    assert [j.name for j in config.jobs] == ["llm-finetune", "vision-cnn", "xgboost-sweep", "tabular-mlp"]
    assert config.jobs[0].device is Device.GPU
    assert config.jobs[2].gpus == 0  # cpu jobs never take GPUs
    assert config.reserve_cpus == 1
    assert config.log_dir == EXAMPLE.parent / "logs"
    assert config.jobs[0].cwd == str(EXAMPLE.parent)


def test_yaml_and_json(tmp_path):
    (tmp_path / "jobs.yaml").write_text(
        "backfill: true\njobs:\n  - name: a\n    command: [python, train.py]\n    device: cpu\n"
    )
    config = load_config(tmp_path / "jobs.yaml")
    assert config.policy.backfill
    assert config.jobs[0].command == ("python", "train.py")

    (tmp_path / "jobs.json").write_text(json.dumps({"jobs": [{"name": "b", "command": "x", "env": {"SEED": 1}}]}))
    config = load_config(tmp_path / "jobs.json")
    assert config.jobs[0].env == {"SEED": "1"}


@pytest.mark.parametrize(
    "data, message",
    [
        ({}, "at least one job"),
        ({"jobs": [{"name": "a", "command": "x", "priorty": 3}]}, "unknown key"),
        ({"jobs": [{"command": "x"}]}, "missing 'name'"),
        ({"jobs": [{"name": "a", "command": "x", "device": "tpu"}]}, "device must be"),
        ({"jobs": [{"name": "a", "command": "x", "priority": 0}]}, "priority"),
        ({"jobs": [{"name": "a", "command": "x", "cpus": 4, "max_cpus": 2}]}, "max_cpus"),
        ({"jobs": [{"name": "bad name", "command": "x"}]}, "invalid job name"),
        ({"jobs": [{"name": "a", "command": ["x"], "shell": True}]}, "single string"),
        ({"resources": {"cores": 4}, "jobs": [{"name": "a", "command": "x"}]}, "unknown key"),
        ({"switch_grace": -1, "jobs": [{"name": "a", "command": "x"}]}, "switch_grace"),
        ({"jobs": [{"name": "a", "command": "x", "max_gpus": "lots"}]}, "max_gpus"),
        ({"jobs": [{"name": "a", "command": "x", "gpu_share": 1.5}]}, "gpu_share"),
        ({"jobs": [{"name": "a", "command": "x", "gpu_memory": "big"}]}, "gpu_memory"),
        ({"resources": {"gpu_memory": "big"}, "jobs": [{"name": "a", "command": "x"}]}, "gpu_memory"),
        ({"resources": {"mps": "yes"}, "jobs": [{"name": "a", "command": "x"}]}, "mps"),
    ],
)
def test_invalid_job_files_are_rejected(data, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


def test_multi_gpu_and_switching_keys():
    config = parse_config({
        "switch_grace": 5, "max_switches": 2,
        "jobs": [{"name": "a", "command": "x", "device": "any", "gpus": 2, "max_gpus": "all", "switchable": True}],
    })
    job = config.jobs[0]
    assert (job.gpus, job.max_gpus, job.switchable) == (2, "all", True)
    assert (config.switch_grace, config.max_switches) == (5.0, 2)


def test_gpu_sharing_keys():
    config = parse_config({
        "resources": {"gpus": "0", "gpu_memory": "24GiB", "mps": True},
        "jobs": [{"name": "a", "command": "x", "gpu_share": 0.5},
                 {"name": "b", "command": "x", "device": "any", "gpu_memory": "8GiB"}],
    })
    assert config.jobs[0].gpu_share == 0.5 and config.jobs[1].gpu_memory == 8 * 1024 ** 3
    assert config.mps and config.inventory().memory_of("0") == 24 * 1024 ** 3


def test_shared_gpu_example_fits_on_one_gpu(capsys):
    from tablely.cli import main

    example = EXAMPLE.with_name("shared_gpu.toml")
    assert [j.gpu_share for j in load_config(example).jobs] == [0.5, None, 0.25]
    assert main(["plan", str(example), "--gpus", "1", "--cpus", "4", "--gpu-memory", "24GiB"]) == 0
    out = capsys.readouterr().out
    assert out.count("start") == 3 and "gpu use: 0 100%" in out
