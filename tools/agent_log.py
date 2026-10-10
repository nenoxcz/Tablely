#!/usr/bin/env python3
"""Work log for coding agents on this repository; the code is in tablely/worklog.py.

Kept as a script so the Claude Code hooks in .claude/settings.json can run it
without Tablely being installed.

    python3 tools/agent_log.py brief | board | task "..." | handoff "..."
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from tablely.worklog import main
except Exception as exc:  # a half-edited checkout must never block the agent's hooks
    if sys.argv[1:2] == ["hook"]:
        print(f"agent_log: {exc}", file=sys.stderr)
        sys.exit(0)
    raise

sys.exit(main())
