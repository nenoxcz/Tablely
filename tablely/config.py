"""Loading job files (TOML, YAML or JSON)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union

from .planner import Policy
from .resources import CpuSetting, GpuSetting, Inventory, build_inventory
from .spec import JobSpec

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

_TOP_KEYS = {"resources", "jobs", "log_dir", "backfill", "task"}
_RESOURCE_KEYS = {"cpus", "gpus", "reserve_cpus"}
_JOB_KEYS = {"name", "command", "priority", "device", "gpus", "cpus", "max_cpus", "env", "cwd", "shell", "task"}


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    jobs: List[JobSpec]
    policy: Policy = field(default_factory=Policy)
    log_dir: Path = Path("tablely-logs")
    cpus: CpuSetting = None
    gpus: GpuSetting = None
    reserve_cpus: int = 0
    task: Optional[str] = None  # what this batch is for; jobs without their own task inherit it

    def inventory(self, simulate: bool = False) -> Inventory:
        try:
            return build_inventory(self.cpus, self.gpus, self.reserve_cpus, simulate=simulate)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"resources: {exc}") from None


def load_config(path: Union[str, Path]) -> Config:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror or exc}") from None
    suffix = path.suffix.lower()
    try:
        if suffix == ".toml":
            data = tomllib.loads(text)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ConfigError("YAML job files need PyYAML: pip install 'tablely[yaml]'") from None
            data = yaml.safe_load(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            raise ConfigError(f"{path}: unsupported file type (use .toml, .yaml or .json)")
    except ConfigError:
        raise
    except Exception as exc:  # parser errors differ per format
        raise ConfigError(f"{path}: {exc}") from None
    return parse_config(data, base_dir=path.parent)


def parse_config(data: Any, base_dir: Union[str, Path] = ".") -> Config:
    base_dir = Path(base_dir).resolve()
    data = _mapping(data, "job file")
    _check_keys(data, _TOP_KEYS, "job file")

    resources = _mapping(data.get("resources", {}), "[resources]")
    _check_keys(resources, _RESOURCE_KEYS, "[resources]")

    raw_jobs = data.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ConfigError("job file must define at least one job under 'jobs'")
    jobs = [_parse_job(raw, i, base_dir) for i, raw in enumerate(raw_jobs, start=1)]

    backfill = data.get("backfill", False)
    if not isinstance(backfill, bool):
        raise ConfigError("backfill must be true or false")

    task = data.get("task")
    if task is not None and not isinstance(task, str):
        raise ConfigError("task must be a string")

    log_dir = Path(str(data.get("log_dir", "tablely-logs")))
    reserve = resources.get("reserve_cpus", 0)
    if isinstance(reserve, bool) or not isinstance(reserve, int):
        raise ConfigError("[resources] reserve_cpus must be an integer")
    return Config(
        jobs=jobs,
        policy=Policy(backfill=backfill),
        log_dir=log_dir if log_dir.is_absolute() else base_dir / log_dir,
        cpus=resources.get("cpus"),
        gpus=resources.get("gpus"),
        reserve_cpus=reserve,
        task=task,
    )


def _parse_job(raw: Any, index: int, base_dir: Path) -> JobSpec:
    where = f"job #{index}"
    raw = _mapping(raw, where)
    if "name" in raw:
        where = f"job {raw['name']!r}"
    _check_keys(raw, _JOB_KEYS, where)
    for required in ("name", "command"):
        if required not in raw:
            raise ConfigError(f"{where}: missing '{required}'")
    fields = dict(raw)
    env = fields.get("env", {})
    if not isinstance(env, Mapping):
        raise ConfigError(f"{where}: env must be a table of NAME = value")
    cwd = Path(str(fields.get("cwd", ".")))
    fields["cwd"] = str(cwd if cwd.is_absolute() else base_dir / cwd)
    if "shell" in fields and not isinstance(fields["shell"], bool):
        raise ConfigError(f"{where}: shell must be true or false")
    try:
        return JobSpec(**fields)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from None


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a table/mapping")
    return value


def _check_keys(data: Mapping[str, Any], allowed: set, where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {', '.join(unknown)} (allowed: {', '.join(sorted(allowed))})"
        )
