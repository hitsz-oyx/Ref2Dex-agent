#!/usr/bin/env python3
"""Resume recent Codex sessions after provider capacity failures.

This is the small, session-only watchdog recovered from the historical
``codex_research_supervisor.py`` implementation. It reads Codex metadata and
rollouts, never creates a child agent, and never uses the retired Broker.

For every unarchived thread updated within the active window, a new
``Selected model is at capacity``/``server_overloaded`` event arms recovery.
When the thread is idle and has no queued input, the watchdog sends ``继续`` at
most once per interval. A later turn start clears the recovery arm; stale
threads naturally expire from the active window.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from typing import Any, Iterable
from urllib.parse import quote
from datetime import datetime, timezone


SCHEMA = "ref2dex.session_capacity_watchdog.v1"
DEFAULT_SCAN_ROOT = "/home2/wyy/oyx_ws"
DEFAULT_ACTIVE_WINDOW = 24 * 60 * 60
DEFAULT_INTERVAL = 60.0
DEFAULT_POLL_INTERVAL = 10.0
CAPACITY_RE = re.compile(
    r"selected model is at capacity|server[_ -]?overloaded|codex_error_info.{0,80}capacity",
    re.IGNORECASE,
)
START_TYPES = frozenset({"task_started", "turn_started", "thread.started"})
END_TYPES = frozenset(
    {
        "task_complete",
        "turn_aborted",
        "turn_failed",
        "turn_cancelled",
        "task_cancelled",
        "task_stopped",
    }
)


def _read_only_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path), safe='/')}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=0.2)


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    except sqlite3.DatabaseError:
        return set()


@dataclass(frozen=True)
class Session:
    home: str
    thread_id: str
    rollout_path: str | None
    archived: bool
    updated_at_ms: int
    subagent: bool = False


@dataclass
class SessionState:
    rollout_path: str | None = None
    inode: int | None = None
    offset: int = 0
    active_turn_id: str | None = None
    capacity_pending: bool = False
    last_capacity_at: float = 0.0
    last_sent_at: float = 0.0


def discover_homes(scan_root: str | Path) -> list[Path]:
    root = Path(scan_root).expanduser().resolve()
    if not root.is_dir():
        return []
    return sorted(
        candidate
        for candidate in root.glob(".codex*")
        if candidate.is_dir() and (candidate / "state_5.sqlite").is_file()
    )


def discover_sessions(home: Path, cutoff_ms: int) -> list[Session]:
    database = home / "state_5.sqlite"
    if not database.is_file():
        return []
    connection: sqlite3.Connection | None = None
    try:
        connection = _read_only_connection(database)
        columns = _table_columns(connection, "threads")
        if not {"id", "rollout_path"}.issubset(columns):
            return []
        updated_column = "updated_at_ms" if "updated_at_ms" in columns else "updated_at"
        select = ["id", "rollout_path", updated_column if updated_column in columns else "NULL"]
        select.append("archived" if "archived" in columns else "0")
        select.append("thread_source" if "thread_source" in columns else "NULL")
        select.append("agent_path" if "agent_path" in columns else "NULL")
        select.append("source" if "source" in columns else "NULL")
        rows = connection.execute(f"SELECT {', '.join(select)} FROM threads").fetchall()
    except (OSError, sqlite3.DatabaseError):
        return []
    finally:
        if connection is not None:
            connection.close()

    sessions: list[Session] = []
    for thread_id, rollout_path, raw_updated, raw_archived, thread_source, agent_path, source in rows:
        if not isinstance(thread_id, str) or not thread_id:
            continue
        try:
            updated = int(raw_updated or 0)
        except (TypeError, ValueError):
            updated = 0
        if updated and updated < 10_000_000_000:
            updated *= 1000
        if not updated and isinstance(rollout_path, str):
            try:
                updated = int(Path(rollout_path).stat().st_mtime * 1000)
            except OSError:
                updated = 0
        if updated < cutoff_ms:
            continue
        sessions.append(
            Session(
                home=str(home),
                thread_id=thread_id,
                rollout_path=rollout_path if isinstance(rollout_path, str) else None,
                archived=bool(raw_archived),
                updated_at_ms=updated,
                subagent=(thread_source == "subagent" or bool(agent_path) or str(source).startswith('{"subagent"')),
            )
        )
    return sessions


def pending_queue_item(home: Path, thread_id: str) -> bool | None:
    database = home / "queue_1.sqlite"
    if not database.is_file():
        return False
    connection: sqlite3.Connection | None = None
    try:
        connection = _read_only_connection(database)
        if not _table_columns(connection, "queued_items"):
            return False
        row = connection.execute(
            "SELECT 1 FROM queued_items WHERE thread_id = ? LIMIT 1", (thread_id,)
        ).fetchone()
        return row is not None
    except (OSError, sqlite3.DatabaseError):
        return None
    finally:
        if connection is not None:
            connection.close()


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else event


def _event_type(event: dict[str, Any]) -> str | None:
    value = _event_payload(event).get("type")
    return value if isinstance(value, str) else None


def _event_turn_id(event: dict[str, Any]) -> str | None:
    payload = _event_payload(event)
    value = payload.get("turn_id") or payload.get("turnId")
    return value if isinstance(value, str) else None


def _event_text(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, sort_keys=True)


def _event_time(event: dict[str, Any], fallback: float) -> float:
    value = event.get("timestamp")
    if not isinstance(value, str):
        value = _event_payload(event).get("timestamp")
    if not isinstance(value, str):
        return fallback
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return fallback


def consume_rollout_events(state: SessionState, rollout: str | None) -> list[dict[str, Any]]:
    """Read complete JSONL records since the saved offset."""

    if not rollout:
        return []
    path = Path(rollout)
    try:
        stat = path.stat()
    except OSError:
        return []
    if state.rollout_path != str(path) or state.inode != stat.st_ino or stat.st_size < state.offset:
        state.rollout_path = str(path)
        state.inode = stat.st_ino
        state.offset = 0
        state.active_turn_id = None
    try:
        with path.open("rb") as handle:
            handle.seek(state.offset)
            chunk = handle.read()
    except OSError:
        return []
    if not chunk:
        return []
    pieces = chunk.split(b"\n")
    tail = pieces.pop()
    state.offset += len(chunk) - len(tail)
    events: list[dict[str, Any]] = []
    for raw in pieces:
        if not raw.strip():
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def apply_events(
    state: SessionState,
    events: Iterable[dict[str, Any]],
    now: float,
    minimum_event_time: float = 0.0,
) -> None:
    for event in events:
        if CAPACITY_RE.search(_event_text(event)) and _event_time(event, now) >= minimum_event_time:
            state.capacity_pending = True
            state.last_capacity_at = now
        kind = _event_type(event)
        turn_id = _event_turn_id(event)
        if kind in START_TYPES:
            state.active_turn_id = turn_id or "unknown"
            # A new turn is evidence that the previous capacity arm was consumed.
            state.capacity_pending = False
        elif kind in END_TYPES and (turn_id is None or turn_id == state.active_turn_id):
            state.active_turn_id = None


class QueueInvoker:
    def __init__(
        self,
        *,
        codex_bin: str | None,
        node: str | None,
        codex_js: str | None,
        timeout: float,
        dry_run: bool,
    ) -> None:
        self.timeout = timeout
        self.dry_run = dry_run
        if dry_run and not (codex_bin or codex_js):
            self.command = []
            return
        if codex_js:
            self.command = [node or shutil.which("node") or "node", codex_js]
        else:
            resolved = codex_bin or shutil.which("codex")
            if not resolved:
                raise ValueError("Codex CLI not found; pass --codex-bin or --codex-js")
            self.command = [resolved]

    def queue(self, home: Path, thread_id: str, message: str) -> bool:
        command = [*self.command, "queue", "--thread", thread_id, "--message", message]
        if self.dry_run:
            return True
        environment = os.environ.copy()
        environment["CODEX_HOME"] = str(home)
        try:
            result = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0


class CapacityWatchdog:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.state_path = Path(args.state)
        self.state: dict[str, Any] = self._load_state()
        self.invoker = QueueInvoker(
            codex_bin=args.codex_bin,
            node=args.node,
            codex_js=args.codex_js,
            timeout=args.command_timeout,
            dry_run=args.dry_run,
        )

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"schema": SCHEMA, "sessions": {}}
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if value.get("schema") != SCHEMA or not isinstance(value.get("sessions"), dict):
                raise ValueError("incompatible watchdog state schema")
            return value
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {"schema": SCHEMA, "sessions": {}}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.state_path.name}.",
            suffix=".tmp",
            dir=str(self.state_path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def cycle(self, *, now: float | None = None) -> dict[str, Any]:
        current = time.time() if now is None else now
        cutoff_ms = int((current - self.args.active_window) * 1000)
        discovered = [
            session
            for home in discover_homes(self.args.scan_root)
            for session in discover_sessions(home, cutoff_ms)
            if not session.archived
        ]
        sessions = [session for session in discovered if not session.subagent]
        live_keys: set[str] = set()
        sent: list[str] = []
        armed: list[str] = []
        defaults = asdict(SessionState())
        for session in sessions:
            key = f"{session.home}::{session.thread_id}"
            live_keys.add(key)
            raw = self.state["sessions"].get(key, {})
            state = SessionState(**{field: raw.get(field, default) for field, default in defaults.items()})
            events = consume_rollout_events(state, session.rollout_path)
            apply_events(state, events, current, current - self.args.active_window)
            queue_status = pending_queue_item(Path(session.home), session.thread_id)
            if state.capacity_pending:
                armed.append(key)
            can_send = (
                state.capacity_pending
                and state.active_turn_id is None
                and queue_status is False
                and current - state.last_sent_at >= self.args.interval
            )
            if can_send and self.invoker.queue(Path(session.home), session.thread_id, self.args.message):
                state.last_sent_at = current
                sent.append(key)
            self.state["sessions"][key] = asdict(state)
        self.state["sessions"] = {
            key: value for key, value in self.state["sessions"].items() if key in live_keys
        }
        self.state["updated_at"] = current
        self._save_state()
        return {
            "scanned": len(discovered),
            "eligible": len(sessions),
            "skipped_subagents": len(discovered) - len(sessions),
            "armed": armed,
            "sent": sent,
        }

    def run(self) -> int:
        while True:
            result = self.cycle()
            if self.args.once:
                print(json.dumps(result, ensure_ascii=False, sort_keys=True))
                return 0
            if result["sent"]:
                print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            time.sleep(self.args.poll_interval)


def _positive(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", default=DEFAULT_SCAN_ROOT)
    parser.add_argument("--state", default=".runtime/session_capacity_watchdog/state.json")
    parser.add_argument("--active-window", type=_positive, default=float(DEFAULT_ACTIVE_WINDOW))
    parser.add_argument("--interval", type=_positive, default=DEFAULT_INTERVAL)
    parser.add_argument("--poll-interval", type=_positive, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--command-timeout", type=_positive, default=30.0)
    parser.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN"))
    parser.add_argument("--node", "--codex-node", dest="node", default=os.environ.get("CODEX_NODE"))
    parser.add_argument("--codex-js", default=os.environ.get("CODEX_CLI_JS"))
    parser.add_argument("--message", default="继续")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run and not (args.codex_bin or args.codex_js):
        parser.error("queue mode requires --codex-bin or --codex-js; use --dry-run for inspection")
    return args


def main(argv: list[str] | None = None) -> int:
    args = build_parser(argv)
    lock_path = Path(args.state + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another session capacity watchdog owns this state file")
        return CapacityWatchdog(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
