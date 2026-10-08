"""Command line: ``tablely resources | plan | run | status | history | note | brief``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import signal
import sys
import time
from typing import List, Optional, Sequence, Union

from . import __version__, _fmt
from .board_view import render_brief, render_history, render_status
from .config import Config, ConfigError, load_config
from .ledger import Board, Ledger, default_agent, make_event
from .planner import Policy, check_feasible, plan
from .resources import Inventory, build_inventory
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

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--home", help="state shared by all agents on this machine (default: $TABLELY_HOME or ~/.tablely)")

    p = sub.add_parser("resources", parents=[resources], help="show the CPUs and GPUs Tablely would use")
    p.set_defaults(func=_cmd_resources)

    p = sub.add_parser(
        "plan",
        parents=[resources, shared],
        help="show how the jobs would be placed right now, around other agents' jobs (dry run)",
    )
    p.add_argument("jobfile")
    p.add_argument("--backfill", action="store_true", default=None, help="let lower-priority jobs use idle resources")
    p.set_defaults(func=_cmd_plan)

    p = sub.add_parser("run", parents=[resources, shared], help="run all jobs to completion")
    p.add_argument("jobfile")
    p.add_argument("--backfill", action="store_true", default=None, help="let lower-priority jobs use idle resources")
    p.add_argument("--log-dir", help="where job output goes (default: from the job file)")
    p.add_argument("--agent", help="who is running this (default: $TABLELY_AGENT or the login name)")
    p.add_argument("--task", help="what this run is for (default: 'task' in the job file)")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("status", parents=[shared], help="who is running what on this machine right now")
    p.add_argument("--json", action="store_true", help="print the raw board as JSON")
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("history", parents=[shared], help="what every agent did: starts, finishes, notes")
    p.add_argument("--agent", help="only this agent")
    p.add_argument("--job", help="only this job name")
    p.add_argument("-n", "--limit", type=int, default=30, help="last N events (default 30, 0 = all)")
    p.add_argument("--json", action="store_true", help="print events as JSON lines")
    p.set_defaults(func=_cmd_history)

    p = sub.add_parser(
        "brief",
        parents=[shared],
        help="handoff for the next agent: what runs now, what finished, what agents noted",
    )
    p.add_argument("--hours", type=float, default=24.0, help="how far back to look (default 24)")
    p.add_argument("--agent", help="only this agent's work")
    p.add_argument("--json", action="store_true", help="print the board and the events as JSON")
    p.set_defaults(func=_cmd_brief)

    p = sub.add_parser("note", parents=[shared], help="record what an agent is working on or what comes next")
    p.add_argument("text", nargs="+")
    p.add_argument("--agent", help="who (default: $TABLELY_AGENT or the login name)")
    p.set_defaults(func=_cmd_note)
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
    # With explicit --cpus/--gpus/--reserve-cpus this is a what-if for some other
    # machine; otherwise plan around whatever other agents are running here.
    what_if = any(v is not None for v in (args.cpus, args.gpus, args.reserve_cpus))
    board = None if what_if else Board(Ledger(args.home).snapshot())
    sharing = board is not None and bool(board.jobs) and board.data.get("pool")
    if sharing:
        pool = board.data["pool"]
        inventory = Inventory(cpus=tuple(pool["cpus"]), gpus=tuple(pool["gpus"]))
        policy = Policy(backfill=pool["backfill"])
    else:
        inventory = config.inventory(simulate=True)  # a preview may describe a bigger server
        policy = config.policy
    errors, warnings = check_feasible(inventory, config.jobs)
    print(f"resources: {inventory.describe()}")
    print(f"policy: {'backfill' if policy.backfill else 'strict priority'}")
    if sharing:
        agents = sorted({job["agent"] for job in board.jobs.values()})
        print(f"planning around {len(board.jobs)} job(s) already here from: {', '.join(agents)}")
    for message in warnings:
        print(f"warning: {message}")
    for message in errors:
        print(f"error: {message}", file=sys.stderr)
    if errors:
        return 2

    specs, running, pending = board.planning_view() if sharing else ({}, {}, [])
    ours = {}
    for spec in config.jobs:
        key = f"new.{spec.name}"
        specs[key] = dataclasses.replace(spec, name=key)
        pending.append(key)
        ours[key] = spec
    decision = plan(inventory, specs, running, pending, policy)
    rows = []
    for key, spec in sorted(ours.items(), key=lambda kv: -kv[1].priority):  # stable: ties keep file order
        alloc = decision.allocations.get(key)
        wants = spec.device.value + (f" x{spec.gpus}" if spec.gpus else "")
        status = "start" if alloc else f"wait: {decision.waiting[key]}"
        rows.append(
            [spec.name, format_priority(spec.priority), wants, _fmt.placement(alloc), _fmt.cores(alloc), status]
        )
    print()
    print(_fmt.table(["JOB", "PRIO", "WANTS", "PLACED ON", "CORES", "STATUS"], rows))
    if sharing:
        rows = []
        for key, job in board.jobs.items():
            alloc = decision.allocations.get(key)
            before = running.get(key)
            if before is None:
                change = "starts" if alloc else f"waits: {decision.waiting.get(key, '')}"
            elif alloc.cpus != before.cpus:
                change = f"cores {_fmt.cores(before)} -> {_fmt.cores(alloc)}"
            else:
                change = "unchanged"
            rows.append([job["agent"], job["name"], format_priority(job["priority"]),
                         _fmt.placement(before), change])
        print()
        print("other agents' jobs:")
        print(_fmt.table(["AGENT", "JOB", "PRIO", "ON", "AFTER YOURS START"], rows))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = _load(args)
    if args.log_dir:
        config = dataclasses.replace(config, log_dir=args.log_dir)
    try:
        runner = Runner(
            config.inventory(),
            config.jobs,
            policy=config.policy,
            log_dir=config.log_dir,
            ledger=Ledger(args.home),
            agent=args.agent,
            task=args.task or config.task,
            switch_grace=config.switch_grace,
            max_switches=config.max_switches,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    signal.signal(signal.SIGTERM, _raise_interrupt)  # `kill tablely` stops jobs cleanly too
    try:
        return runner.run()
    except ValueError as exc:  # does not fit the pool other agents already share
        raise ConfigError(str(exc)) from None


def _cmd_status(args: argparse.Namespace) -> int:
    data = Ledger(args.home).snapshot()
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(render_status(data))
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    events = Ledger(args.home).history(agent=args.agent, job=args.job, limit=args.limit or None)
    if args.json:
        for event in events:
            print(json.dumps(event, ensure_ascii=False))
    else:
        print(render_history(events))
    return 0


def _cmd_brief(args: argparse.Namespace) -> int:
    ledger = Ledger(args.home)
    data = ledger.snapshot()
    since = time.time() - args.hours * 3600
    events = [e for e in ledger.history(agent=args.agent) if e.get("t", 0) >= since]
    if args.json:
        print(json.dumps({"board": data, "events": events}, indent=2, ensure_ascii=False))
    else:
        print(render_brief(data, events, hours=args.hours, agent=args.agent))
    return 0


def _cmd_note(args: argparse.Namespace) -> int:
    agent = args.agent or default_agent()
    text = " ".join(args.text)
    with Ledger(args.home).locked() as board:
        board.set_note(agent, text)
        board.log(make_event(time.time(), "note", agent, None, None, None, text))
    print(f"noted for {agent}: {text}")
    return 0


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
