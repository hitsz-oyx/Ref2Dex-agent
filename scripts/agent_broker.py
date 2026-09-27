#!/usr/bin/env python3
"""Deterministic task broker for the fixed Ref2Dex worker pool.

The broker is deliberately provider agnostic and contains no model or research
policy.  ``root`` submits a task to a stable ``agent_key``; a worker claims it,
emits updates, and writes a handoff.  Provider adapters only translate a
dispatch into a transport-neutral request for an already configured runtime.
They do not create agents or share credentials.

The command line is useful for smoke checks and for a small external adapter
daemon.  Runtime state is local and ignored by Git by design.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MESSAGE_TYPES = frozenset({"TASK_DISPATCH", "TASK_UPDATE", "TASK_HANDOFF", "CONTROL"})
TASK_STATES = frozenset({
    "PENDING", "LEASED", "RUNNING", "HANDOFF_READY", "COMPLETED", "FAILED",
    "CANCELLED",
})
CONTROL_ACTIONS = frozenset({"PAUSE", "RESUME", "CANCEL", "STOP", "RELOAD", "WAKE", "BUDGET_LIMITED"})
SCHEMA = "ref2dex.agent_broker.v1"
ROLES_SCHEMA = "ref2dex.agent_roles.v1"


def _now() -> float:
    return time.time()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _object(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)


def _read_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def load_roles(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the tracked role file without importing runtime bindings."""

    try:
        import yaml  # type: ignore
    except ImportError as error:  # pragma: no cover - dependency is optional
        raise RuntimeError("PyYAML is required to read AGENT_ROLES.yaml") from error
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != ROLES_SCHEMA:
        raise ValueError("invalid AGENT_ROLES schema")
    roles = value.get("roles")
    if not isinstance(roles, list):
        raise ValueError("AGENT_ROLES.roles must be a list")
    result: dict[str, dict[str, Any]] = {}
    for role in roles:
        if not isinstance(role, dict) or not isinstance(role.get("agent_key"), str):
            raise ValueError("each role needs an agent_key")
        key = role["agent_key"]
        if key in result:
            raise ValueError(f"duplicate agent_key: {key}")
        result[key] = role
    if "root" not in result:
        raise ValueError("fixed role pool must include root")
    return result


def load_bindings(path: str | Path) -> dict[str, dict[str, Any]]:
    candidate = Path(path)
    if not candidate.is_file():
        return {}
    value = _read_json(candidate)
    bindings = value.get("bindings", {})
    if not isinstance(bindings, dict):
        raise ValueError("AGENT_BINDINGS.bindings must be an object")
    return {str(key): item for key, item in bindings.items() if isinstance(item, dict)}


class BrokerError(RuntimeError):
    """A caller-visible broker contract error."""


@dataclass(frozen=True)
class RuntimeRequest:
    provider: str
    runtime_agent_key: str
    task_id: str
    payload: dict[str, Any]


class ProviderAdapter:
    """Translate broker work into a request for a configured external runtime."""

    provider = "generic"

    def status(self, binding: Mapping[str, Any]) -> dict[str, Any]:
        """Return transport metadata; the external runtime owns live status."""

        return {"provider": self.provider, "runtime_agent_key": binding.get("runtime_agent_key"),
                "status": "external"}

    def control(self, action: str, binding: Mapping[str, Any]) -> dict[str, Any]:
        """Build a bounded control request without contacting a provider."""

        return {"type": "CONTROL", "action": action, "runtime_agent_key": binding.get("runtime_agent_key")}

    def dispatch(self, task: Mapping[str, Any], binding: Mapping[str, Any]) -> RuntimeRequest:
        runtime_key = binding.get("runtime_agent_key")
        if not isinstance(runtime_key, str) or not runtime_key:
            raise BrokerError("target role has no runtime binding")
        return RuntimeRequest(
            provider=self.provider,
            runtime_agent_key=runtime_key,
            task_id=str(task["task_id"]),
            payload={
                "type": "TASK_DISPATCH",
                "task_id": task["task_id"],
                "target": task["target_agent"],
                "objective": task["objective"],
                "context": task["context"],
                "constraints": task["constraints"],
                "done_when": task["done_when"],
            },
        )


class CodexAppServerAdapter(ProviderAdapter):
    provider = "codex_app_server"


class CodexCliAdapter(ProviderAdapter):
    provider = "codex_cli"


class NewApiAdapter(ProviderAdapter):
    provider = "newapi"


class GenericAdapter(ProviderAdapter):
    provider = "other"


# Names used by the design document remain available to external adapter
# daemons; the broker itself only depends on the small ProviderAdapter API.
CodexAppServerRuntime = CodexAppServerAdapter
CodexCliRuntime = CodexCliAdapter
NewApiRuntime = NewApiAdapter
OtherProviderRuntime = GenericAdapter


ADAPTERS = {
    "codex_app_server": CodexAppServerAdapter,
    "codex_cli": CodexCliAdapter,
    "newapi": NewApiAdapter,
    "other": GenericAdapter,
}


def adapter_for(binding: Mapping[str, Any]) -> ProviderAdapter:
    provider = binding.get("provider") or binding.get("runtime") or "other"
    try:
        return ADAPTERS[str(provider)]()
    except KeyError as error:
        raise BrokerError(f"unsupported provider adapter: {provider}") from error


class AgentBroker:
    """SQLite-backed queue, lease and handoff store."""

    def __init__(
        self,
        tasks_db: str | Path = ".runtime/tasks.sqlite",
        state_db: str | Path = ".runtime/AGENT_STATE.sqlite",
        roles_path: str | Path = "docs/AGENT_ROLES.yaml",
        bindings_path: str | Path = ".runtime/AGENT_BINDINGS.json",
    ) -> None:
        self.tasks_db = Path(tasks_db)
        self.state_db = Path(state_db)
        self.roles_path = Path(roles_path)
        self.bindings_path = Path(bindings_path)
        self.roles = load_roles(self.roles_path)
        self.bindings = load_bindings(self.bindings_path)
        self.tasks_db.parent.mkdir(parents=True, exist_ok=True)
        self.state_db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self, path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _init_db(self) -> None:
        with self._connect(self.tasks_db) as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                  task_id TEXT PRIMARY KEY,
                  target_agent TEXT NOT NULL,
                  objective TEXT NOT NULL,
                  context_json TEXT NOT NULL,
                  constraints_json TEXT NOT NULL,
                  done_when_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  lease_token TEXT,
                  lease_expires_at REAL,
                  attempts INTEGER NOT NULL DEFAULT 0,
                  last_error TEXT
                );
                CREATE TABLE IF NOT EXISTS messages (
                  message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  type TEXT NOT NULL CHECK(type IN ('TASK_DISPATCH','TASK_UPDATE','TASK_HANDOFF','CONTROL')),
                  task_id TEXT,
                  sender TEXT NOT NULL,
                  recipient TEXT,
                  payload_json TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  delivered_at REAL
                );
                CREATE INDEX IF NOT EXISTS messages_task_idx ON messages(task_id, message_id);
                CREATE TABLE IF NOT EXISTS handoffs (
                  task_id TEXT PRIMARY KEY,
                  agent_key TEXT NOT NULL,
                  status TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at REAL NOT NULL
                );
                """
            )
        with self._connect(self.state_db) as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS runtime_state (
                  agent_key TEXT PRIMARY KEY,
                  status TEXT NOT NULL,
                  current_task TEXT,
                  last_event TEXT,
                  last_error TEXT,
                  heartbeat_at REAL,
                  updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS supervisor_state (
                  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                  desired_state TEXT NOT NULL,
                  generation INTEGER NOT NULL,
                  reason TEXT,
                  updated_at REAL NOT NULL
                );
                INSERT OR IGNORE INTO supervisor_state(singleton, desired_state, generation, reason, updated_at)
                  VALUES (1, 'PAUSED', 0, 'initial state', 0);
                """
            )

    def _role(self, agent_key: str) -> dict[str, Any]:
        try:
            return self.roles[agent_key]
        except KeyError as error:
            raise BrokerError(f"unknown fixed agent_key: {agent_key}") from error

    def _binding(self, agent_key: str) -> dict[str, Any]:
        self._role(agent_key)
        binding = self.bindings.get(agent_key, {})
        if not isinstance(binding, dict):
            raise BrokerError(f"invalid binding for {agent_key}")
        return binding

    @staticmethod
    def _row_task(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for key in ("context_json", "constraints_json", "done_when_json"):
            result[key[:-5]] = _object(result.pop(key), {})
        return result

    def _message(
        self,
        db: sqlite3.Connection,
        message_type: str,
        *,
        task_id: str | None,
        sender: str,
        recipient: str | None,
        payload: Mapping[str, Any],
    ) -> int:
        if message_type not in MESSAGE_TYPES:
            raise BrokerError(f"unsupported message type: {message_type}")
        cursor = db.execute(
            "INSERT INTO messages(type, task_id, sender, recipient, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (message_type, task_id, sender, recipient, _json(dict(payload)), _now()),
        )
        return int(cursor.lastrowid)

    def dispatch(
        self,
        *,
        task_id: str,
        target_agent: str,
        objective: str,
        context: Mapping[str, Any] | None = None,
        constraints: Mapping[str, Any] | None = None,
        done_when: list[str] | None = None,
        sender: str = "root",
    ) -> dict[str, Any]:
        if not task_id or not objective.strip():
            raise BrokerError("task_id and objective are required")
        role = self._role(target_agent)
        if target_agent == "root":
            raise BrokerError("root is the decision maker; dispatch only to a worker role")
        if role.get("lifecycle", "active") != "active":
            raise BrokerError(f"target role is not active: {target_agent}")
        binding = self._binding(target_agent)
        task = {
            "task_id": task_id,
            "target_agent": target_agent,
            "objective": objective.strip(),
            "context": dict(context or {}),
            "constraints": dict(constraints or {}),
            "done_when": list(done_when or []),
        }
        adapter = adapter_for(binding)
        request = adapter.dispatch(task, binding)
        now = _now()
        with self._connect(self.tasks_db) as db:
            try:
                db.execute(
                    "INSERT INTO tasks(task_id,target_agent,objective,context_json,constraints_json,done_when_json,status,created_at,updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)",
                    (task_id, target_agent, task["objective"], _json(task["context"]),
                     _json(task["constraints"]), _json(task["done_when"]), now, now),
                )
            except sqlite3.IntegrityError as error:
                raise BrokerError(f"task already exists: {task_id}") from error
            message_id = self._message(
                db, "TASK_DISPATCH", task_id=task_id, sender=sender,
                recipient=target_agent,
                payload={**task, "provider_request": request.payload, "provider": request.provider},
            )
        self.update_runtime(target_agent, status="DISPATCHED", current_task=task_id, event="TASK_DISPATCH")
        return {"task_id": task_id, "message_id": message_id, "provider": request.provider, "runtime_agent_key": request.runtime_agent_key}

    def claim(self, agent_key: str, lease_seconds: float = 900.0) -> dict[str, Any] | None:
        self._role(agent_key)
        if agent_key == "root":
            raise BrokerError("root cannot claim worker tasks")
        if lease_seconds <= 0:
            raise BrokerError("lease_seconds must be positive")
        now = _now()
        with self._connect(self.tasks_db) as db:
            db.execute("BEGIN IMMEDIATE")
            # A dead adapter must not strand a fixed worker forever.  The
            # expired lease returns to PENDING and is counted on the next
            # claim; no watchdog or model is needed for this recovery.
            db.execute(
                "UPDATE tasks SET status='PENDING', lease_token=NULL, lease_expires_at=NULL, updated_at=? "
                "WHERE status='LEASED' AND lease_expires_at IS NOT NULL AND lease_expires_at<=?",
                (now, now),
            )
            active = db.execute(
                "SELECT task_id FROM tasks WHERE target_agent=? AND status IN ('LEASED','RUNNING','HANDOFF_READY') LIMIT 1",
                (agent_key,),
            ).fetchone()
            if active is not None:
                db.commit()
                return None
            row = db.execute(
                "SELECT * FROM tasks WHERE target_agent=? AND status='PENDING' ORDER BY created_at, task_id LIMIT 1",
                (agent_key,),
            ).fetchone()
            if row is None:
                db.commit()
                return None
            token = secrets.token_urlsafe(18)
            expires = now + lease_seconds
            db.execute(
                "UPDATE tasks SET status='LEASED', lease_token=?, lease_expires_at=?, attempts=attempts+1, updated_at=? WHERE task_id=?",
                (token, expires, now, row["task_id"]),
            )
            db.commit()
            task = self._row_task(row)
            task.update({"status": "LEASED", "lease_token": token, "lease_expires_at": expires})
        self.update_runtime(agent_key, status="LEASED", current_task=task["task_id"], event="TASK_CLAIM")
        return task

    def update(
        self,
        *,
        task_id: str,
        agent_key: str,
        status: str = "RUNNING",
        progress: Mapping[str, Any] | None = None,
        lease_token: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        self._role(agent_key)
        status = status.upper()
        if status not in TASK_STATES:
            raise BrokerError(f"invalid task status: {status}")
        now = _now()
        payload = {"status": status, "progress": dict(progress or {})}
        with self._connect(self.tasks_db) as db:
            row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise BrokerError(f"unknown task: {task_id}")
            if row["target_agent"] != agent_key:
                raise BrokerError("task update must come from its target agent")
            if lease_token is not None and row["lease_token"] != lease_token:
                raise BrokerError("invalid task lease token")
            db.execute(
                "UPDATE tasks SET status=?, updated_at=?, last_error=? WHERE task_id=?",
                (status, now, error, task_id),
            )
            message_id = self._message(
                db, "TASK_UPDATE", task_id=task_id, sender=agent_key, recipient="root", payload=payload,
            )
        self.update_runtime(agent_key, status=status, current_task=task_id, event="TASK_UPDATE", error=error)
        return {"task_id": task_id, "message_id": message_id, **payload}

    def handoff(
        self,
        *,
        task_id: str,
        agent_key: str,
        status: str,
        result: Mapping[str, Any] | None = None,
        evidence: list[str] | None = None,
        commit: str | None = None,
        recommended_next_action: str | None = None,
        lease_token: str | None = None,
    ) -> dict[str, Any]:
        self._role(agent_key)
        status = status.upper()
        if status not in {"COMPLETED", "FAILED", "CANCELLED", "HANDOFF_READY"}:
            raise BrokerError("handoff status must be COMPLETED, FAILED, CANCELLED or HANDOFF_READY")
        payload = {
            "status": status,
            "result": dict(result or {}),
            "evidence": list(evidence or []),
            "commit": commit,
            "recommended_next_action": recommended_next_action,
        }
        now = _now()
        with self._connect(self.tasks_db) as db:
            row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise BrokerError(f"unknown task: {task_id}")
            if row["target_agent"] != agent_key:
                raise BrokerError("handoff must come from its target agent")
            if lease_token is not None and row["lease_token"] != lease_token:
                raise BrokerError("invalid task lease token")
            db.execute("UPDATE tasks SET status=?, updated_at=? WHERE task_id=?", (status, now, task_id))
            db.execute(
                "INSERT OR REPLACE INTO handoffs(task_id,agent_key,status,payload_json,created_at) VALUES (?, ?, ?, ?, ?)",
                (task_id, agent_key, status, _json(payload), now),
            )
            message_id = self._message(
                db, "TASK_HANDOFF", task_id=task_id, sender=agent_key, recipient="root", payload=payload,
            )
        self.update_runtime(agent_key, status=status, current_task=None if status != "HANDOFF_READY" else task_id,
                            event="TASK_HANDOFF")
        return {"task_id": task_id, "message_id": message_id, **payload}

    def control(self, *, action: str, target: str = "root", reason: str = "", sender: str = "root") -> dict[str, Any]:
        action = action.upper()
        if action not in CONTROL_ACTIONS:
            raise BrokerError(f"unsupported control action: {action}")
        self._role(target)
        payload = {"action": action, "target": target, "reason": reason}
        with self._connect(self.tasks_db) as db:
            message_id = self._message(db, "CONTROL", task_id=None, sender=sender, recipient=target, payload=payload)
        if target == "root" and action in {"PAUSE", "RESUME"}:
            self.set_desired_state("PAUSED" if action == "PAUSE" else "RUNNING", reason=reason)
        return {"message_id": message_id, **payload}

    def observe(self, *, agent_key: str, payload: Mapping[str, Any], task_id: str | None = None,
                sender: str = "worker_event_poller") -> dict[str, Any]:
        """Persist a liveness/progress observation as a normal TASK_UPDATE.

        The event poller is an observation source, not a second protocol.  A
        task id is optional because a worker can change branch/heartbeat while
        between tasks; the root still receives one typed broker message.
        """

        self._role(agent_key)
        if agent_key == "root":
            raise BrokerError("worker observations cannot target root")
        value = {"status": "OBSERVED", "observation": dict(payload)}
        with self._connect(self.tasks_db) as db:
            message_id = self._message(
                db, "TASK_UPDATE", task_id=task_id, sender=sender, recipient="root", payload=value,
            )
        self.update_runtime(agent_key, status="OBSERVED", current_task=task_id, event="TASK_UPDATE")
        return {"message_id": message_id, "agent_key": agent_key, **value}

    def update_runtime(self, agent_key: str, *, status: str, current_task: str | None,
                       event: str, error: str | None = None) -> None:
        self._role(agent_key)
        now = _now()
        with self._connect(self.state_db) as db:
            db.execute(
                "INSERT INTO runtime_state(agent_key,status,current_task,last_event,last_error,heartbeat_at,updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(agent_key) DO UPDATE SET status=excluded.status,current_task=excluded.current_task,"
                "last_event=excluded.last_event,last_error=excluded.last_error,heartbeat_at=excluded.heartbeat_at,updated_at=excluded.updated_at",
                (agent_key, status, current_task, event, error, now, now),
            )

    def set_desired_state(self, desired_state: str, *, reason: str = "") -> dict[str, Any]:
        desired_state = desired_state.upper()
        if desired_state not in {"RUNNING", "PAUSED", "STOPPED"}:
            raise BrokerError("desired_state must be RUNNING, PAUSED or STOPPED")
        now = _now()
        with self._connect(self.state_db) as db:
            row = db.execute("SELECT generation FROM supervisor_state WHERE singleton=1").fetchone()
            generation = int(row[0] if row else 0) + 1
            db.execute(
                "INSERT INTO supervisor_state(singleton,desired_state,generation,reason,updated_at) VALUES (1, ?, ?, ?, ?) "
                "ON CONFLICT(singleton) DO UPDATE SET desired_state=excluded.desired_state,generation=excluded.generation,reason=excluded.reason,updated_at=excluded.updated_at",
                (desired_state, generation, reason, now),
            )
        return {"desired_state": desired_state, "generation": generation, "reason": reason, "updated_at": now}

    def status(self) -> dict[str, Any]:
        with self._connect(self.tasks_db) as db:
            tasks = [dict(row) for row in db.execute("SELECT * FROM tasks ORDER BY created_at, task_id")]
            messages = [dict(row) for row in db.execute("SELECT * FROM messages ORDER BY message_id DESC LIMIT 50")]
            handoffs = [dict(row) for row in db.execute("SELECT * FROM handoffs ORDER BY created_at DESC")]
        for task in tasks:
            for key in ("context_json", "constraints_json", "done_when_json"):
                task[key[:-5]] = _object(task.pop(key), {})
        for message in messages:
            message["payload"] = _object(message.pop("payload_json"), {})
        with self._connect(self.state_db) as db:
            supervisor = db.execute("SELECT * FROM supervisor_state WHERE singleton=1").fetchone()
            runtime = [dict(row) for row in db.execute("SELECT * FROM runtime_state ORDER BY agent_key")]
        return {
            "schema": SCHEMA,
            "supervisor": dict(supervisor) if supervisor else None,
            "runtime": runtime,
            "tasks": tasks,
            "messages": messages,
            "handoffs": handoffs,
        }

    def messages_since(self, message_id: int = 0) -> list[dict[str, Any]]:
        with self._connect(self.tasks_db) as db:
            rows = db.execute("SELECT * FROM messages WHERE message_id>? ORDER BY message_id", (message_id,)).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            value["payload"] = _object(value.pop("payload_json"), {})
            result.append(value)
        return result


def _broker_from_args(args: argparse.Namespace) -> AgentBroker:
    return AgentBroker(args.tasks_db, args.state_db, args.roles, args.bindings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-db", default=".runtime/tasks.sqlite")
    parser.add_argument("--state-db", default=".runtime/AGENT_STATE.sqlite")
    parser.add_argument("--roles", default="docs/AGENT_ROLES.yaml")
    parser.add_argument("--bindings", default=".runtime/AGENT_BINDINGS.json")
    sub = parser.add_subparsers(dest="command", required=True)
    dispatch = sub.add_parser("dispatch")
    dispatch.add_argument("--task-id", required=True)
    dispatch.add_argument("--target", required=True)
    dispatch.add_argument("--objective", required=True)
    dispatch.add_argument("--context", default="{}")
    dispatch.add_argument("--constraints", default="{}")
    dispatch.add_argument("--done-when", action="append", default=[])
    claim = sub.add_parser("claim")
    claim.add_argument("--agent", required=True)
    claim.add_argument("--lease-seconds", type=float, default=900.0)
    update = sub.add_parser("update")
    update.add_argument("--task-id", required=True)
    update.add_argument("--agent", required=True)
    update.add_argument("--status", default="RUNNING")
    update.add_argument("--progress", default="{}")
    update.add_argument("--lease-token")
    handoff = sub.add_parser("handoff")
    handoff.add_argument("--task-id", required=True)
    handoff.add_argument("--agent", required=True)
    handoff.add_argument("--status", required=True)
    handoff.add_argument("--result", default="{}")
    handoff.add_argument("--evidence", action="append", default=[])
    handoff.add_argument("--commit")
    handoff.add_argument("--next")
    handoff.add_argument("--lease-token")
    control = sub.add_parser("control")
    control.add_argument("--action", required=True)
    control.add_argument("--target", default="root")
    control.add_argument("--reason", default="")
    status = sub.add_parser("status")
    status.add_argument("--since", type=int)
    args = parser.parse_args(argv)
    try:
        broker = _broker_from_args(args)
        if args.command == "dispatch":
            value = broker.dispatch(task_id=args.task_id, target_agent=args.target, objective=args.objective,
                                    context=json.loads(args.context), constraints=json.loads(args.constraints),
                                    done_when=args.done_when)
        elif args.command == "claim":
            value = broker.claim(args.agent, args.lease_seconds)
        elif args.command == "update":
            value = broker.update(task_id=args.task_id, agent_key=args.agent, status=args.status,
                                  progress=json.loads(args.progress), lease_token=args.lease_token)
        elif args.command == "handoff":
            value = broker.handoff(task_id=args.task_id, agent_key=args.agent, status=args.status,
                                   result=json.loads(args.result), evidence=args.evidence, commit=args.commit,
                                   recommended_next_action=args.next, lease_token=args.lease_token)
        elif args.command == "control":
            value = broker.control(action=args.action, target=args.target, reason=args.reason)
        else:
            value = broker.messages_since(args.since) if args.since is not None else broker.status()
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (BrokerError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
