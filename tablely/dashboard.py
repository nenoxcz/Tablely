"""``tablely ui``: a local web dashboard of who is doing what, and how far along.

* Training: every run (live, or finished within ``hours``) with its completion
  rate and each job's progress bar; per-agent completion and latest note.
* Coding: when pointed at a repository, its coding-agent sessions from the work
  log, with each session's task list as a completion bar.
* Every card has a Resume button. Tablely writes a summary of that work (the
  same text as ``tablely resume``), saves it, and either shows it to copy into
  an AI, or — with ``--resume-command`` such as ``"claude -p"`` — starts the AI
  with it right away.

Only the standard library is used. The server listens on 127.0.0.1 by default.
Resume requests need a token that is only embedded in the page this server
serves, and requests for other host names are refused, so other web pages
cannot trigger a launch.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

from . import worklog
from ._dashboard_page import PAGE
from .ledger import Ledger
from .progress import agent_achievement, collect_runs, hand_to_ai, resume_prompt, session_prompt

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


class Dashboard:
    """What the page shows and what Resume does; independent of HTTP for testing."""

    def __init__(
        self,
        ledger: Ledger,
        repo: Optional[str] = None,
        hours: float = 72.0,
        resume_command: Optional[str] = None,
    ) -> None:
        self.ledger = ledger
        self.repo = repo
        self.hours = hours
        self.resume_command = resume_command
        self.token = secrets.token_urlsafe(24)

    # -- state ------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        now = time.time()
        since = now - self.hours * 3600
        data = self.ledger.snapshot()
        events = self.ledger.history()
        runs = [r for r in collect_runs(data, events) if r.live or (r.ended or 0) >= since]
        notes = [e for e in events if e.get("event") == "note" and e.get("t", 0) >= since][-20:][::-1]
        latest_note = {}
        for note in notes:
            latest_note.setdefault(note.get("agent"), note)
        agents = []
        for name in dict.fromkeys(r.agent for r in runs):
            mine = [r for r in runs if r.agent == name]
            agents.append({
                "name": name,
                "achievement": agent_achievement(mine),
                "live_runs": sum(r.live for r in mine),
                "runs": len(mine),
                "note": latest_note.get(name),
            })
        return {
            "generated_at": now,
            "hours": self.hours,
            "pool": data.get("pool"),
            "agents": agents,
            "runs": [r.to_json() for r in runs],
            "notes": notes,
            "sessions": self._sessions(),
            "repo": self.repo,
            "resume_command": self.resume_command,
        }

    def _sessions(self) -> List[Dict[str, Any]]:
        if not self.repo:
            return []
        out = []
        for s in worklog.collect(worklog.repo_root(self.repo)):
            status = worklog.shown_status(s)
            if status == "ended" and worklog.age_seconds(s.get("updated_at")) > self.hours * 3600:
                continue
            done, total = worklog.todo_counts(s)
            out.append({
                "session": s["session"],
                "short": s["session"][:8],
                "branch": s.get("branch"),
                "status": status,
                "task": s.get("task"),
                "updated_at": s.get("updated_at"),
                "updated_ago": worklog.ago(s.get("updated_at")),
                "todos": s.get("todos") or [],
                "done": done,
                "total": total,
                "achievement": done / total if total else None,
                "handoff": (s.get("handoff") or {}).get("text"),
                "last_reply": (s.get("last_reply") or {}).get("text"),
                "files": list(reversed(s.get("files") or []))[:8],
                "seen_in": s.get("seen_in"),
            })
        return out

    # -- resume -----------------------------------------------------------------

    def resume(self, kind: str, ident: str) -> Tuple[int, Dict[str, Any]]:
        """Build the summary for one card and hand it over. Returns (HTTP status, body)."""
        now = time.time()
        cwd = None
        if kind == "session":
            session = worklog.find_session(self.repo, ident) if self.repo else None
            if session is None:
                return HTTPStatus.NOT_FOUND, {"error": f"no coding session {ident!r}"}
            prompt, label, cwd = session_prompt(session), f"session-{session['session'][:8]}", self.repo
        elif kind in ("run", "agent"):
            events = self.ledger.history()
            since = now - self.hours * 3600
            runs = [
                r for r in collect_runs(self.ledger.snapshot(), events)
                if (r.run == ident if kind == "run" else r.agent == ident)
                and (kind == "run" or r.live or (r.ended or 0) >= since)
            ]
            if not runs:
                return HTTPStatus.NOT_FOUND, {"error": f"no {kind} {ident!r}"}
            agent = runs[0].agent
            notes = [e for e in events if e.get("event") == "note" and e.get("agent") == agent
                     and e.get("t", 0) >= since][-10:][::-1]
            prompt, label = resume_prompt(runs, notes, agent=agent), f"{kind}-{ident}"
        else:
            return HTTPStatus.BAD_REQUEST, {"error": f"unknown kind {kind!r}"}
        result = hand_to_ai(prompt, self.ledger.home / "resume", label, command=self.resume_command, cwd=cwd)
        return HTTPStatus.OK, result


def make_server(dashboard: Dashboard, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    page = PAGE.replace("__TOKEN__", dashboard.token).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        server_version = "tablely-ui"

        def log_message(self, fmt: str, *args: Any) -> None:  # keep the terminal quiet
            pass

        def _host_ok(self) -> bool:
            if host not in LOCAL_HOSTS:
                return True  # explicitly exposed (e.g. behind an SSH tunnel or proxy)
            name = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            return name in LOCAL_HOSTS

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: Any) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "unexpected Host"})
            if self.path in ("/", "/index.html"):
                return self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self._json(HTTPStatus.OK, dashboard.state())
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "unexpected Host"})
            if self.path != "/api/resume":
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            if not secrets.compare_digest(self.headers.get("X-Tablely-Token", ""), dashboard.token):
                return self._json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong token"})
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 64 * 1024)
                body = json.loads(self.rfile.read(length) or b"{}")
                kind, ident = str(body["kind"]), str(body["id"])
            except (ValueError, KeyError, TypeError):
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "expected JSON {kind, id}"})
            status, result = dashboard.resume(kind, ident)
            self._json(status, result)

    return ThreadingHTTPServer((host, port), Handler)


def serve(dashboard: Dashboard, host: str, port: int, out: Any = None) -> None:
    server = make_server(dashboard, host, port)
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"Tablely dashboard: http://{shown}:{server.server_address[1]}/  (Ctrl+C to stop)", file=out, flush=True)
    if dashboard.resume_command:
        print(f"Resume starts: {dashboard.resume_command} <summary>", file=out, flush=True)
    else:
        print("Resume shows the summary to copy; set --resume-command to start an AI with it", file=out, flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def serve_in_thread(dashboard: Dashboard, host: str = "127.0.0.1", port: int = 0) -> Tuple[ThreadingHTTPServer, str]:
    """Start the server in the background (tests, notebooks). Returns (server, base URL)."""
    server = make_server(dashboard, host, port)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    return server, f"http://{host}:{server.server_address[1]}"
