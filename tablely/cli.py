"""Command line: ``tablely resources | plan | run``."""

from __future__ import annotations

import argparse
import dataclasses
import re
import signal
import sys
from typing import List, Optional, Sequence, Union

from . import __version__, _fmt
from .config import Config, ConfigError, load_config
from .planner import Policy, check_feasible, plan
from .resources import build_inventory
from .runner import Runner
from .spec import format_priority


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"tablely: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tablely",
        description="Run several training jobs on one machine, sharing GPUs and CPU cores by priority.",
    )
    parser.add_argument("--version", action="version", version=f"tablely {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    resources = argparse.ArgumentParser(add_help=False)
    resources.add_argument("--cpus", type=_cpu_arg, help="core count (e.g. 12) or core list (e.g. 0-7,16-23)")
    resources.add_argument("--gpus", type=_gpu_arg, help="GPU count (e.g. 2) or id list (e.g. 0,2)")
    resources.add_argument("--reserve-cpus", type=int, help="leave the lowest N cores to the OS")

    p = sub.add_parser("resources", parents=[resources], help="show the CPUs and GPUs Tablely would use")
    p.set_defaults(func=_cmd_resources)

    p = sub.add_parser("plan", parents=[resources], help="show how the jobs would be placed right now (dry run)")
    p.add_argument("jobfile")
    p.add_argument("--backfill", action="store_true", default=None, help="let lower-priority jobs use idle resources")
    p.set_defaults(func=_cmd_plan)

    p = sub.add_parser("run", parents=[resources], help="run all jobs to completion")
    p.add_argument("jobfile")
    p.add_argument("--backfill", action="store_true", default=None, help="let lower-priority jobs use idle resources")
    p.add_argument("--log-dir", help="where job output goes (default: from the job file)")
    p.set_defaults(func=_cmd_run)
    return parser


def _cmd_resources(args: argparse.Namespace) -> int:
    try:
        inventory = build_inventory(args.cpus, args.gpus, args.reserve_cpus or 0)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    print(inventory.describe())
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    config = _load(args)
    inventory = config.inventory(simulate=True)  # a preview may describe a bigger server
    errors, warnings = check_feasible(inventory, config.jobs)
    print(f"resources: {inventory.describe()}")
    print(f"policy: {'backfill' if config.policy.backfill else 'strict priority'}")
    for message in warnings:
        print(f"warning: {message}")
    for message in errors:
        print(f"error: {message}", file=sys.stderr)
    if errors:
        return 2

    specs = {spec.name: spec for spec in config.jobs}
    decision = plan(inventory, specs, {}, list(specs), config.policy)
    rows = []
    for spec in sorted(config.jobs, key=lambda s: -s.priority):  # stable: ties keep file order
        alloc = decision.allocations.get(spec.name)
        wants = spec.device.value + (f" x{spec.gpus}" if spec.gpus else "")
        status = "start" if alloc else f"wait: {decision.waiting[spec.name]}"
        rows.append(
            [spec.name, format_priority(spec.priority), wants, _fmt.placement(alloc), _fmt.cores(alloc), status]
        )
    print()
    print(_fmt.table(["JOB", "PRIO", "WANTS", "PLACED ON", "CORES", "STATUS"], rows))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = _load(args)
    if args.log_dir:
        config = dataclasses.replace(config, log_dir=args.log_dir)
    try:
        runner = Runner(config.inventory(), config.jobs, policy=config.policy, log_dir=config.log_dir)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    signal.signal(signal.SIGTERM, _raise_interrupt)  # `kill tablely` stops jobs cleanly too
    return runner.run()


def _load(args: argparse.Namespace) -> Config:
    config = load_config(args.jobfile)
    overrides = {}
    if args.cpus is not None:
        overrides["cpus"] = args.cpus
    if args.gpus is not None:
        overrides["gpus"] = args.gpus
    if args.reserve_cpus is not None:
        overrides["reserve_cpus"] = args.reserve_cpus
    if args.backfill:
        overrides["policy"] = Policy(backfill=True)
    return dataclasses.replace(config, **overrides)


def _cpu_arg(text: str) -> Union[int, str]:
    return int(text) if text.isdigit() else text


def _gpu_arg(text: str) -> Union[int, List[str]]:
    if text.isdigit():
        return int(text)
    if not re.fullmatch(r"[\w-]+(,[\w-]+)*", text):
        raise argparse.ArgumentTypeError(f"expected a count or comma-separated ids, got {text!r}")
    return text.split(",")


def _raise_interrupt(signum, frame):  # noqa: ARG001
    raise KeyboardInterrupt


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
