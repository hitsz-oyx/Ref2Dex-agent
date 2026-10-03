#!/usr/bin/env python3
# DRAIN_ONLY: legacy workflow; new tasks use researchctl and docs/workflow/ARCHITECTURE.md.
"""Retired compatibility wrapper for the former fixed-message wake-up daemon.

The research workflow now requires ``/root`` to poll in its own active turn and
dispatch explicit child goals only when an authorized blocker exists.  This
module remains only for historical audits and tests.  Real queueing is refused
unless the caller opts into the deprecated ``--legacy-fixed-message`` mode.

Historical dry-run syntax (does not queue a message)::

    python3 scripts/codex_research_supervisor.py \
      --thread 01a0d943-de74-7021-8d50-2a4e87fde613 \
      --codex-home /home2/wyy/oyx_ws/.codex_oyx_NewAPI \
      --node /home2/wyy/.nvm/versions/node/v24.19.0/bin/node \
      --codex-js /home2/wyy/.nvm/versions/node/v24.19.0/lib/node_modules/@openai/codex/bin/codex.js \
      --interval 300 --start-immediately --allow-blocked --dry-run --once

Use ``--dry-run --once`` only to inspect legacy eligibility.  Do not launch a
long-running instance.  The supervisor persists only its own offsets and wake-
up bookkeeping in a small JSON file, and reads Codex's SQLite databases in
read-only mode.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote


LOGGER = logging.getLogger("codex-research-supervisor")
DEFAULT_INTERVAL = 300.0
DEFAULT_POLL_INTERVAL = 30.0
DEFAULT_ACK_TIMEOUT = 180.0
DEFAULT_COMMAND_TIMEOUT = 60.0
ROOT_START = "task_started"
ROOT_TERMINALS = frozenset(
    {
        "task_complete",
        "turn_aborted",
        "turn_failed",
        "turn_cancelled",
        "task_cancelled",
        "task_stopped",
    }
)
THREAD_RE = re.compile(r"[A-Za-z0-9_-]+$")


def safe_thread_id(value: str) -> str:
    if not value or not THREAD_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "thread must contain only letters, digits, underscore, or hyphen"
        )
    return value


def _read_only_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path), safe='/')}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=0.2)


@dataclass
class ThreadSnapshot:
    rollout_path: Optional[Path]
    archived: Optional[bool]
    title: Optional[str]
    cwd: Optional[str]
    goal_status: Optional[str]


def _find_rollout(codex_home: Path, thread_id: str) -> Optional[Path]:
    state_db = codex_home / "state_5.sqlite"
    if state_db.exists():
        connection: Optional[sqlite3.Connection] = None
        try:
            connection = _read_only_connection(state_db)
            row = connection.execute(
                "SELECT rollout_path FROM threads WHERE id = ?", (thread_id,)
            ).fetchone()
            if row and isinstance(row[0], str):
                candidate = Path(row[0])
                if candidate.exists():
                    return candidate
        except (OSError, sqlite3.DatabaseError) as exc:
            LOGGER.warning("cannot read %s: %s", state_db, exc)
        finally:
            if connection is not None:
                connection.close()

    sessions = codex_home / "sessions"
    if not sessions.exists():
        return None
    candidates = list(sessions.glob(f"*/*/*/rollout-*-{thread_id}.jsonl"))
    candidates.sort(key=lambda item: item.stat().st_mtime_ns, reverse=True)
    return candidates[0] if candidates else None


def _goal_status(codex_home: Path, thread_id: str) -> Optional[str]:
    database = codex_home / "goals_1.sqlite"
    if not database.exists():
        return None
    connection: Optional[sqlite3.Connection] = None
    try:
        connection = _read_only_connection(database)
        row = connection.execute(
            "SELECT status FROM thread_goals WHERE thread_id = ? LIMIT 1",
            (thread_id,),
        ).fetchone()
        return row[0] if row and isinstance(row[0], str) else None
    except (OSError, sqlite3.DatabaseError) as exc:
        LOGGER.warning("cannot read Goal status from %s: %s", database, exc)
        return None
    finally:
        if connection is not None:
            connection.close()


def read_thread_snapshot(codex_home: Path, thread_id: str) -> ThreadSnapshot:
    state_db = codex_home / "state_5.sqlite"
    rollout = _find_rollout(codex_home, thread_id)
    archived: Optional[bool] = None
    title: Optional[str] = None
    cwd: Optional[str] = None
    connection: Optional[sqlite3.Connection] = None
    if state_db.exists():
        try:
            connection = _read_only_connection(state_db)
            row = connection.execute(
                "SELECT archived, title, cwd, rollout_path FROM threads WHERE id = ?",
                (thread_id,),
            ).fetchone()
            if row:
                archived = bool(row[0])
                title = row[1] if isinstance(row[1], str) else None
                cwd = row[2] if isinstance(row[2], str) else None
                if rollout is None and isinstance(row[3], str) and Path(row[3]).exists():
                    rollout = Path(row[3])
        except (OSError, sqlite3.DatabaseError) as exc:
            LOGGER.warning("cannot read thread metadata from %s: %s", state_db, exc)
        finally:
            if connection is not None:
                connection.close()
    return ThreadSnapshot(rollout, archived, title, cwd, _goal_status(codex_home, thread_id))


def pending_queue_item(codex_home: Path, thread_id: str) -> Optional[bool]:
    """Return True/False, or None when the queue cannot be inspected safely."""

    database = codex_home / "queue_1.sqlite"
    if not database.exists():
        return False
    connection: Optional[sqlite3.Connection] = None
    try:
        connection = _read_only_connection(database)
        row = connection.execute(
            "SELECT 1 FROM queued_items WHERE thread_id = ? LIMIT 1", (thread_id,)
        ).fetchone()
        return row is not None
    except (OSError, sqlite3.DatabaseError) as exc:
        LOGGER.warning("cannot inspect queue %s: %s", database, exc)
        return None
    finally:
        if connection is not None:
            connection.close()


def _payload(event: Dict[str, Any]) -> Dict[str, Any]:
    value = event.get("payload")
    return value if isinstance(value, dict) else {}


def _is_root_start(payload: Dict[str, Any]) -> bool:
    turn_id = payload.get("turn_id")
    root_turn_id = payload.get("root_turn_id")
    return isinstance(turn_id, str) and (
        not root_turn_id or root_turn_id == turn_id
    )


def apply_activity_event(
    active_turn_id: Optional[str], event: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str]]:
    """Apply one rollout event and return ``(active_turn, effect)``.

    ``effect`` is ``root_started``, ``root_finished`` or None.  Child-agent
    events are ignored so a child completing cannot make the parent look idle.
    """

    payload = _payload(event)
    event_type = payload.get("type")
    turn_id = payload.get("turn_id")
    root_turn_id = payload.get("root_turn_id")
    if event_type == ROOT_START and _is_root_start(payload):
        return turn_id, "root_started"
    if event_type in ROOT_TERMINALS and isinstance(turn_id, str):
        if turn_id == active_turn_id or root_turn_id == turn_id:
            return None, "root_finished"
    return active_turn_id, None


@dataclass
class SupervisorState:
    thread_id: str
    rollout_path: Optional[str] = None
    inode: Optional[int] = None
    offset: int = 0
    active_turn_id: Optional[str] = None
    last_wake_at: float = 0.0
    wake_count: int = 0
    waiting_for_turn: bool = False
    started_at: float = 0.0

    @classmethod
    def load(cls, path: Path, thread_id: str) -> "SupervisorState":
        if not path.exists():
            return cls(thread_id=thread_id, started_at=time.time())
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("thread_id") != thread_id:
                raise ValueError("state file belongs to another thread")
            return cls(
                thread_id=thread_id,
                rollout_path=document.get("rollout_path"),
                inode=document.get("inode"),
                offset=int(document.get("offset", 0) or 0),
                active_turn_id=document.get("active_turn_id"),
                last_wake_at=float(document.get("last_wake_at", 0.0) or 0.0),
                wake_count=int(document.get("wake_count", 0) or 0),
                waiting_for_turn=bool(document.get("waiting_for_turn", False)),
                started_at=float(document.get("started_at", time.time()) or time.time()),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            LOGGER.warning("cannot read supervisor state %s: %s; resetting", path, exc)
            return cls(thread_id=thread_id, started_at=time.time())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "version": 1,
            "updated_at": time.time(),
            "thread_id": self.thread_id,
            "rollout_path": self.rollout_path,
            "inode": self.inode,
            "offset": self.offset,
            "active_turn_id": self.active_turn_id,
            "last_wake_at": self.last_wake_at,
            "wake_count": self.wake_count,
            "waiting_for_turn": self.waiting_for_turn,
            "started_at": self.started_at,
        }
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def consume_rollout_events(
    state: SupervisorState, rollout: Optional[Path]
) -> List[Dict[str, Any]]:
    """Read complete JSONL records since the saved offset."""

    if rollout is None:
        return []
    try:
        stat = rollout.stat()
    except OSError:
        return []
    changed = state.rollout_path != str(rollout) or state.inode != stat.st_ino
    if changed or stat.st_size < state.offset:
        state.rollout_path = str(rollout)
        state.inode = stat.st_ino
        state.offset = 0
        state.active_turn_id = None
    try:
        with rollout.open("rb") as handle:
            handle.seek(state.offset)
            chunk = handle.read()
    except OSError as exc:
        LOGGER.warning("cannot read rollout %s: %s", rollout, exc)
        return []
    if not chunk:
        return []
    pieces = chunk.split(b"\n")
    tail = pieces.pop()
    state.offset += len(chunk) - len(tail)
    events: List[Dict[str, Any]] = []
    for raw in pieces:
        if not raw.strip():
            continue
        try:
            event = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            LOGGER.warning("ignoring malformed rollout record: %s", exc)
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


class QueueInvoker:
    def __init__(
        self,
        codex_home: Path,
        *,
        codex_bin: Optional[str],
        node: Optional[str],
        codex_js: Optional[str],
        timeout: float,
        dry_run: bool,
    ) -> None:
        self.codex_home = codex_home
        self.timeout = timeout
        self.dry_run = dry_run
        if codex_js:
            self.command = [node or shutil.which("node") or "node", codex_js]
        else:
            resolved = codex_bin or shutil.which("codex")
            if not resolved:
                raise ValueError("Codex CLI not found; pass --codex-bin or --codex-js")
            self.command = [resolved]

    def queue(self, thread_id: str, message: str) -> bool:
        command = self.command + ["queue", "--thread", thread_id, "--message", message]
        if self.dry_run:
            LOGGER.info("dry-run: would execute %s", " ".join(command))
            return True
        environment = os.environ.copy()
        environment["CODEX_HOME"] = str(self.codex_home)
        try:
            result = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            LOGGER.warning("queue command failed: %s", exc)
            return False
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            LOGGER.warning("queue command returned %s%s", result.returncode, f": {details}" if details else "")
            return False
        LOGGER.info("queued supervisor message for %s", thread_id)
        return True


class ProcessLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Optional[Any] = None

    def __enter__(self) -> "ProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"another supervisor already holds {self.path}") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()}\nstarted_at={time.time():.3f}\n")
        self.handle.flush()
        return self

    def __exit__(self, *_: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class ResearchSupervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop_requested = False
        self.state = SupervisorState.load(args.state_file, args.thread)
        self.invoker: Optional[QueueInvoker] = None
        if args.dry_run or args.legacy_fixed_message:
            self.invoker = QueueInvoker(
                args.codex_home,
                codex_bin=args.codex_bin,
                node=args.node,
                codex_js=args.codex_js,
                timeout=args.command_timeout,
                dry_run=args.dry_run,
            )

    def stop(self, *_: Any) -> None:
        self.stop_requested = True

    def _observe(self, snapshot: ThreadSnapshot) -> List[Dict[str, Any]]:
        events = consume_rollout_events(self.state, snapshot.rollout_path)
        for event in events:
            self.state.active_turn_id, effect = apply_activity_event(
                self.state.active_turn_id, event
            )
            if self.state.waiting_for_turn and effect in {"root_started", "root_finished"}:
                self.state.waiting_for_turn = False
        return events

    def _eligible(self, snapshot: ThreadSnapshot, now: float) -> Tuple[bool, str]:
        if snapshot.archived is True:
            return False, "thread is archived"
        if snapshot.rollout_path is None:
            return False, "rollout path is unavailable"
        if self.state.active_turn_id:
            return False, f"turn {self.state.active_turn_id} is active"
        if snapshot.goal_status in {"complete", "paused"}:
            return False, f"Goal status is {snapshot.goal_status}"
        if snapshot.goal_status == "blocked" and not self.args.allow_blocked:
            return False, "Goal is blocked; pass --allow-blocked to opt in"
        queued = pending_queue_item(self.args.codex_home, self.args.thread)
        if queued is None:
            return False, "queue state is unavailable"
        if queued:
            return False, "a message is already queued"
        if self.state.waiting_for_turn:
            if now - self.state.last_wake_at < self.args.ack_timeout:
                return False, "waiting for the previous wake to start a turn"
            LOGGER.warning("previous wake was not acknowledged; allowing a retry")
            self.state.waiting_for_turn = False
        if self.state.last_wake_at == 0.0:
            if not self.args.start_immediately:
                self.state.last_wake_at = now
                return False, "initial interval has not elapsed"
        elif now - self.state.last_wake_at < self.args.interval:
            return False, "wake interval has not elapsed"
        if self.args.max_wakes and self.state.wake_count >= self.args.max_wakes:
            return False, "max-wakes reached"
        return True, "ready"

    def cycle(self, *, now: Optional[float] = None) -> bool:
        current = time.time() if now is None else now
        snapshot = read_thread_snapshot(self.args.codex_home, self.args.thread)
        self._observe(snapshot)
        eligible, reason = self._eligible(snapshot, current)
        if not eligible:
            LOGGER.debug("not waking %s: %s", self.args.thread, reason)
            self.state.save(self.args.state_file)
            return False
        if self.args.dry_run:
            LOGGER.info("dry-run: thread %s is ready to receive a supervisor wake", self.args.thread)
            self.state.save(self.args.state_file)
            return False
        if not self.args.legacy_fixed_message:
            LOGGER.warning(
                "legacy fixed-message queueing is retired; use /root in-turn polling"
            )
            self.state.save(self.args.state_file)
            return False
        if self.invoker is None or not self.invoker.queue(self.args.thread, self.args.message):
            self.state.save(self.args.state_file)
            return False
        self.state.last_wake_at = current
        self.state.wake_count += 1
        self.state.waiting_for_turn = True
        self.state.save(self.args.state_file)
        return True

    def run(self) -> int:
        if not self.args.dry_run and not self.args.legacy_fixed_message:
            LOGGER.warning(
                "research supervisor is retired; no message will be queued. "
                "Use /root's in-turn polling and explicit GOAL_DISPATCH."
            )
            self.state.save(self.args.state_file)
            return 0
        LOGGER.info(
            "supervising thread %s every %.1fs (poll %.1fs)",
            self.args.thread,
            self.args.interval,
            self.args.poll_interval,
        )
        while not self.stop_requested:
            if self.args.max_runtime and time.time() - self.state.started_at >= self.args.max_runtime:
                LOGGER.info("max-runtime reached; stopping")
                break
            self.cycle()
            if self.args.once:
                break
            time.sleep(self.args.poll_interval)
        self.state.save(self.args.state_file)
        return 0


def _positive(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread", required=True, type=safe_thread_id)
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")),
    )
    parser.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN"))
    parser.add_argument("--node", default=os.environ.get("CODEX_NODE"))
    parser.add_argument("--codex-js", default=os.environ.get("CODEX_CLI_JS"))
    parser.add_argument("--interval", type=_positive, default=DEFAULT_INTERVAL)
    parser.add_argument("--poll-interval", type=_positive, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--ack-timeout", type=_positive, default=DEFAULT_ACK_TIMEOUT)
    parser.add_argument("--command-timeout", type=_positive, default=DEFAULT_COMMAND_TIMEOUT)
    parser.add_argument("--max-runtime", type=_positive, default=0.0)
    parser.add_argument("--max-wakes", type=int, default=0)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--lock-file", type=Path, default=None)
    parser.add_argument(
        "--message",
        default=None,
        help="explicit message for deprecated legacy mode; never used by the normal workflow",
    )
    parser.add_argument(
        "--legacy-fixed-message",
        action="store_true",
        help="explicitly opt into the retired fixed-message queue mode",
    )
    parser.add_argument("--start-immediately", action="store_true")
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="allow waking a thread whose persisted Goal status is blocked",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.max_wakes < 0:
        parser.error("--max-wakes cannot be negative")
    if args.legacy_fixed_message and not args.message:
        parser.error("--legacy-fixed-message requires an explicit --message")
    if args.state_file is None:
        args.state_file = repo_root / "outputs" / "codex_research_supervisor" / f"{args.thread}.json"
    if args.lock_file is None:
        args.lock_file = args.state_file.with_suffix(".lock")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    supervisor = ResearchSupervisor(args)
    signal.signal(signal.SIGINT, supervisor.stop)
    signal.signal(signal.SIGTERM, supervisor.stop)
    try:
        with ProcessLock(args.lock_file):
            return supervisor.run()
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
