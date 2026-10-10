"""Command line: ``tablely resources | plan | run | status | history | brief | resume | mcp | note``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

from . import __version__, _fmt, handoff
from .board_view import render_brief, render_history
from .config import Config, ConfigError, load_config
from .ledger import Board, Ledger, default_agent, make_event
from .planner import Policy, check_feasible, plan
from .progress import run_ai, save_prompt
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

    p = sub.add_parser("status", parents=[shared], help="progress of everyone's work on this machine")
    p.add_argument("--watch", nargs="?", const=2.0, type=float, metavar="SECONDS",
                   help="keep refreshing (every 2s, or SECONDS) until Ctrl+C")
    p.add_argument("--hours", type=float, default=12.0, help="show runs that finished this recently (default 12)")
    p.add_argument("--repo", default=".", help="also show coding sessions of this repository (default: .)")
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

    p = sub.add_parser(
        "resume",
        parents=[shared],
        help="pick earlier work, summarize it and start an AI CLI with the summary",
    )
    p.add_argument("target", nargs="?",
                   help="agent name, run id or coding-session id (a prefix is enough); omit to pick from a list")
    p.add_argument("--repo", default=".", help="repository whose coding sessions can be resumed (default: .)")
    p.add_argument("--hours", type=float, default=72.0, help="how far back to offer finished work (default 72)")
    p.add_argument("--ai", metavar="COMMAND",
                   help='AI CLI to start, e.g. "claude" (default: $TABLELY_AI, else claude if installed); '
                        "{prompt_file} in it is replaced by the saved summary's path")
    p.add_argument("--print", dest="print_only", action="store_true",
                   help="only print the summary, e.g. to pipe it: tablely resume claude-a --print | claude -p")
    p.add_argument("-y", "--yes", action="store_true", help="start the AI without asking")
    p.set_defaults(func=_cmd_resume)

    p = sub.add_parser(
        "mcp",
        parents=[shared],
        help="let AI apps refer to Tablely over MCP (stdio for Claude Desktop/Code; --http for remote connectors)",
    )
    p.add_argument("--http", action="store_true",
                   help="serve HTTP instead of stdio, for the Claude or ChatGPT app through an HTTPS tunnel")
    p.add_argument("--host", default="127.0.0.1", help="--http address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8766, help="--http port (default 8766)")
    p.add_argument("--token", help="--http secret (default: $TABLELY_MCP_TOKEN, else kept in $TABLELY_HOME/mcp-token)")
    p.add_argument("--repo", default=None,
                   help="repository whose coding sessions to include (default: the current directory if it is one)")
    p.set_defaults(func=_cmd_mcp)

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
    ledger = Ledger(args.home)
    if args.json:
        print(json.dumps(ledger.snapshot(), indent=2, ensure_ascii=False))
        return 0
    color = _fmt.use_color()

    def frame() -> str:
        return handoff.status_text(ledger, args.repo, args.hours, color=color)

    if args.watch is None:
        print(frame())
        return 0
    try:
        while True:
            sys.stdout.write("\033[H\033[2J" + frame()
                             + f"\n\nrefreshing every {args.watch:g}s · Ctrl+C to stop\n")
            sys.stdout.flush()
            time.sleep(max(args.watch, 0.2))
    except KeyboardInterrupt:
        print()
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


def _cmd_resume(args: argparse.Namespace) -> int:
    ledger = Ledger(args.home)
    runs, sessions = handoff.resumable(ledger, args.repo, args.hours)
    try:
        if args.target:
            choice = handoff.match(args.target, runs, args.repo)
        else:
            options = handoff.choices(runs, sessions)
            if not options:
                raise ConfigError(f"nothing to resume: no runs in the last {args.hours:g}h and no coding sessions here")
            if not _interactive():
                print("\n".join(c.line for c in options))
                raise ConfigError("name one to resume: tablely resume <id>")
            choice = _pick(options)
            if choice is None:
                return 0
        prompt, label, cwd = handoff.summarize(ledger, choice, runs, args.repo, args.hours)
    except LookupError as exc:
        raise ConfigError(f"{exc} (see: tablely resume)") from None
    path = save_prompt(prompt, ledger.home / "resume", label)

    if args.print_only:
        print(prompt)
        return 0
    print(prompt)
    print(_fmt.paint(f"\n(saved to {path})", "dim", _fmt.use_color()))
    command = args.ai or os.environ.get("TABLELY_AI") or ("claude" if shutil.which("claude") else None)
    if command is None:
        print("No AI CLI found. Paste the summary above into your AI, or set TABLELY_AI / pass --ai \"<command>\".")
        return 0
    if not args.yes:
        if not _interactive():
            print(f"To hand it over: tablely resume {choice.ident} --ai {shlex.quote(command)} --yes")
            return 0
        answer = input(f"Start `{command}` with this summary? [Y/n] ").strip().lower()
        if answer not in ("", "y", "yes"):
            return 0
    try:
        return run_ai(command, prompt, path, cwd=cwd)
    except RuntimeError as exc:
        raise ConfigError(str(exc)) from None


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _pick(choices: List[handoff.Choice]) -> Optional[handoff.Choice]:
    print("Resume which work?")
    for i, choice in enumerate(choices, start=1):
        print(f"  {i:>2}  {choice.line}")
    while True:
        answer = input(f"number [1-{len(choices)}, Enter = 1, q = quit]: ").strip().lower()
        if answer in ("q", "quit"):
            return None
        if answer == "":
            return choices[0]
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        print("  not a number from the list")


def _cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp import McpServer, load_token, make_http_server, serve_stdio
    from .worklog import git

    ledger = Ledger(args.home)
    repo = args.repo or (os.getcwd() if git(Path.cwd(), "rev-parse", "--show-toplevel") else None)
    server = McpServer(ledger, repo)
    if not args.http:
        serve_stdio(server)  # stdout belongs to the protocol from here on
        return 0
    token = load_token(ledger.home, args.token)
    try:
        httpd = make_http_server(server, args.host, args.port, token)
    except OSError as exc:
        raise ConfigError(f"cannot listen on {args.host}:{args.port}: {exc.strerror or exc}") from None
    port = httpd.server_address[1]
    print(f"Tablely MCP server: http://{args.host}:{port}/mcp/{token}", file=sys.stderr)
    print("For the Claude or ChatGPT app, expose it over HTTPS (e.g. cloudflared tunnel --url "
          f"http://127.0.0.1:{port}) and add https://<tunnel>/mcp/{token} as a connector. Keep the URL secret.",
          file=sys.stderr, flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
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
