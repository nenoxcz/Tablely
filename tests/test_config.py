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
