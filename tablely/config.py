"""Loading job files (TOML, YAML or JSON)."""

from __future__ import annotations

import itertools
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .planner import Policy
from .resources import CpuSetting, GpuSetting, Inventory, build_inventory, parse_bytes
from .spec import JobSpec

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

_TOP_KEYS = {"resources", "jobs", "log_dir", "backfill", "task", "switch_grace", "max_switches"}
_RESOURCE_KEYS = {"cpus", "gpus", "reserve_cpus", "gpu_memory", "mps", "ram"}
_JOB_KEYS = {
    "name", "command", "priority", "device", "gpus", "max_gpus", "cpus", "max_cpus",
    "env", "cwd", "shell", "task", "switchable", "gpu_share", "gpu_memory", "ram",
    "after", "retries", "timeout", "matrix", "cpu_command",
}
_RUNTIME_PLACEHOLDERS = {"name", "device", "gpus", "num_gpus", "cpus", "cpu_list"}
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
    switch_grace: float = 60.0  # seconds a switchable job runs before it may be asked to move
    max_switches: int = 5  # device switches per job before it is no longer asked
    gpu_memory: Union[None, int, str] = None  # memory of each GPU, when nvidia-smi cannot tell
    mps: bool = False  # an NVIDIA MPS daemon runs: let the driver enforce GPU shares too
    ram: Union[None, int, str] = None  # RAM jobs may reserve for tables (default: half the machine's)

    def inventory(self, simulate: bool = False) -> Inventory:
        try:
            return build_inventory(self.cpus, self.gpus, self.reserve_cpus, simulate=simulate,
                                   gpu_memory=self.gpu_memory, ram=self.ram)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"resources: {exc}") from None


def load_config(path: Union[str, Path]) -> Config:
    path = Path(path)
    return parse_config(read_file(path), base_dir=path.parent)


def read_file(path: Union[str, Path]) -> Any:
    """The data in a TOML, YAML or JSON file."""
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
    return data


def parse_config(data: Any, base_dir: Union[str, Path] = ".") -> Config:
    base_dir = Path(base_dir).resolve()
    data = _mapping(data, "job file")
    _check_keys(data, _TOP_KEYS, "job file")

    resources = _mapping(data.get("resources", {}), "[resources]")
    _check_keys(resources, _RESOURCE_KEYS, "[resources]")

    raw_jobs = data.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ConfigError("job file must define at least one job under 'jobs'")

    task = data.get("task")
    if task is not None and not isinstance(task, str):
        raise ConfigError("task must be a string")

    expanded: List[Tuple[int, Mapping[str, Any]]] = []
    groups: Dict[str, List[str]] = {}  # a swept job's name -> the names of its variants
    for i, raw in enumerate(raw_jobs, start=1):
        variants = _expand_matrix(raw, i, task)
        if len(variants) > 1 or "matrix" in _mapping(raw, f"job #{i}"):
            groups[str(raw.get("name"))] = [v["name"] for v in variants]
        expanded += [(i, v) for v in variants]
    jobs = [_parse_job(_resolve_after(raw, groups), i, base_dir) for i, raw in expanded]

    backfill = data.get("backfill", False)
    if not isinstance(backfill, bool):
        raise ConfigError("backfill must be true or false")

    switch_grace = data.get("switch_grace", 60.0)
    if isinstance(switch_grace, bool) or not isinstance(switch_grace, (int, float)) or switch_grace < 0:
        raise ConfigError("switch_grace must be a non-negative number of seconds")
    max_switches = data.get("max_switches", 5)
    if isinstance(max_switches, bool) or not isinstance(max_switches, int) or max_switches < 0:
        raise ConfigError("max_switches must be a non-negative integer")

    for key in ("gpu_memory", "ram"):
        if resources.get(key) is not None:
            try:
                parse_bytes(resources[key])
            except ValueError as exc:
                raise ConfigError(f"[resources] {key}: {exc}") from None
    mps = resources.get("mps", False)
    if not isinstance(mps, bool):
        raise ConfigError("[resources] mps must be true or false")

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
        switch_grace=float(switch_grace),
        max_switches=max_switches,
        gpu_memory=resources.get("gpu_memory"),
        mps=mps,
        ram=resources.get("ram"),
    )


def _expand_matrix(raw: Any, index: int, run_task: Optional[str]) -> List[Dict[str, Any]]:
    """One job per combination of ``matrix`` values, with ``{key}`` filled in.

    ``matrix = {lr = [1e-3, 3e-4], seed = [0, 1]}`` on job ``train`` gives
    ``train.lr-0.001.seed-0`` and three more; ``{lr}`` and ``{seed}`` in the
    command, env values, cwd and task are replaced by each variant's values.
    """
    raw = dict(_mapping(raw, f"job #{index}"))
    matrix = raw.pop("matrix", None)
    if matrix is None:
        return [raw]
    where = f"job {raw.get('name', f'#{index}')!r}"
    matrix = _mapping(matrix, f"{where}: matrix")
    if not matrix:
        raise ConfigError(f"{where}: matrix is empty")
    for key, values in matrix.items():
        if not _KEY_RE.match(key) or key in _RUNTIME_PLACEHOLDERS:
            raise ConfigError(f"{where}: matrix key {key!r} must be a plain name other than "
                              f"{', '.join(sorted(_RUNTIME_PLACEHOLDERS))}")
        if not isinstance(values, list) or not values or not all(
            isinstance(v, (str, int, float, bool)) for v in values
        ):
            raise ConfigError(f"{where}: matrix {key} must be a non-empty list of values")
    keys = list(matrix)
    pattern = re.compile(r"\{(" + "|".join(map(re.escape, keys)) + r")\}")
    variants = []
    for combo in itertools.product(*(matrix[k] for k in keys)):
        values = {k: _text(v) for k, v in zip(keys, combo)}

        def fill(text: Any) -> Any:
            return pattern.sub(lambda m: values[m.group(1)], text) if isinstance(text, str) else text

        job = dict(raw)
        job["name"] = ".".join([str(raw.get("name"))] + [f"{k}-{_slug(values[k])}" for k in keys])
        for key in ("command", "cpu_command"):
            if key in raw:
                value = raw[key]
                job[key] = [fill(part) for part in value] if isinstance(value, list) else fill(value)
        if isinstance(raw.get("env"), Mapping):
            job["env"] = {k: fill(v) for k, v in raw["env"].items()}
        if "cwd" in raw:
            job["cwd"] = fill(raw["cwd"])
        label = ", ".join(f"{k}={values[k]}" for k in keys)
        base = fill(raw.get("task")) if raw.get("task") else run_task
        job["task"] = f"{base} ({label})" if base else label
        variants.append(job)
    return variants


def _text(value: Any) -> str:
    return ("true" if value else "false") if isinstance(value, bool) else str(value)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text) or "_"


def _resolve_after(raw: Mapping[str, Any], groups: Mapping[str, List[str]]) -> Mapping[str, Any]:
    """``after = ["train"]`` on a swept job ``train`` means after every variant of it."""
    after = raw.get("after")
    if after is None:
        return raw
    names = [after] if isinstance(after, str) else after
    if not isinstance(names, list):
        raise ConfigError(f"job {raw.get('name')!r}: after must be a list of job names")
    resolved: List[Any] = []
    for name in names:
        resolved += groups.get(name, [name]) if isinstance(name, str) else [name]
    return dict(raw, after=resolved)


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
