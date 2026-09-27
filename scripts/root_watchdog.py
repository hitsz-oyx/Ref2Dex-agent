#!/usr/bin/env python3
"""Keep the root supervision turn live through the Agent Broker.

With broker databases configured, the watchdog writes ``CONTROL`` messages and
its own JSON state; a lease-authorized ``paused`` or ``blocked`` root Goal is
also resumed through the registered root app-server and verified by readback.
It never reads experiment metrics, starts a worker, or selects a research route.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

try:  # Works both as ``python -m scripts...`` and a direct script path.
    from scripts import agent_result_poller as runtime
    from scripts.agent_broker import AgentBroker
except ModuleNotFoundError:  # pragma: no cover - direct CLI entry point
    import agent_result_poller as runtime
    from agent_broker import AgentBroker


SCHEMA = "ref2dex.root_watchdog.v1"
LEASE_SCHEMA = "ref2dex.supervisor_lease.v1"
ROOT_LIVENESS_WAKE = "ROOT_LIVENESS_WAKE\n"
ROOT_BUDGET_LIMITED = "ROOT_BUDGET_LIMITED\n"
ACTIVE = frozenset({"active", "running"})
BUDGET = frozenset({"usage_limited", "budget_limited"})
TERMINAL = frozenset({"complete", "completed", "failed", "stopped", "terminated"})
RESUMABLE = frozenset({"paused", "blocked"})
ROOT_IDENTITY_FIELDS = ("conversation_id", "codex_home", "worktree", "branch")
ROOT_BINDING_SCHEMA = "ref2dex.agent_bindings.v2"


def load_lease(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_file():
        return {
            "schema": LEASE_SCHEMA,
            "enabled": False,
            "mode": "manual",
            "root_agent": "root",
            "allow_root_resume": False,
            "generation": 0,
        }
    value = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") not in {None, LEASE_SCHEMA}:
        raise ValueError("invalid supervisor lease schema")
    return value


def _registry_root(registry: dict) -> dict | None:
    for agent in registry.get("agents", []):
        if isinstance(agent, dict) and agent.get("agent_key") == "root":
            return agent
    return None


def root_runtime_identity(registry_path: str | Path,
                          registry: dict | None = None) -> tuple[dict | None, str]:
    """Resolve the active root identity from the v2 machine-local binding."""

    path = Path(registry_path)
    try:
        registry = registry if isinstance(registry, dict) else runtime.read_json(path)
    except (OSError, ValueError, TypeError) as error:
        return None, f"registry unreadable: {error}"
    root = _registry_root(registry)
    if root is None:
        return None, "registry has no root runtime"
    metadata = registry.get("runtime_bindings")
    binding_path = metadata.get("path") if isinstance(metadata, dict) else None
    if not isinstance(binding_path, str) or not binding_path:
        return None, "runtime binding path missing"
    binding_path = Path(binding_path)
    if not binding_path.is_absolute():
        binding_path = path.resolve().parent.parent / binding_path
    try:
        document = runtime.read_json(binding_path)
    except (OSError, ValueError, TypeError) as error:
        return None, f"runtime binding unreadable: {error}"
    if not isinstance(document, dict) or document.get("schema") != ROOT_BINDING_SCHEMA:
        return None, "runtime binding schema is not v2"
    bindings = document.get("bindings")
    binding = bindings.get("root") if isinstance(bindings, dict) else None
    if not isinstance(binding, dict) or binding.get("status") != "bound":
        return None, "root runtime binding missing or unbound"
    if binding.get("runtime_agent_key") != "root":
        return None, "root runtime binding selects a different agent"
    identity = {}
    for field in ROOT_IDENTITY_FIELDS:
        expected = root.get(field)
        selected = binding.get(field)
        if not isinstance(expected, str) or not expected:
            return None, f"registry root {field} missing"
        if not isinstance(selected, str) or not selected:
            return None, f"runtime binding {field} missing"
        if selected != expected:
            return None, f"runtime binding {field} mismatches registry root"
        identity[field] = expected
    binding_provider = binding.get("provider")
    registry_provider = root.get("provider")
    if registry_provider is not None or binding_provider is not None:
        if not isinstance(binding_provider, str) or not binding_provider:
            return None, "runtime binding provider missing"
        if registry_provider is not None and registry_provider != binding_provider:
            return None, "runtime binding provider mismatches registry root"
        identity["provider"] = binding_provider
    identity["runtime_agent_key"] = "root"
    identity["role_key"] = "root"
    return identity, "matched"


def lease_identity_matches(lease: dict, identity: dict | None) -> tuple[bool, str]:
    """Require lease identity to match the selected root runtime exactly."""

    if identity is None:
        return False, "root runtime identity unavailable"
    if not isinstance(lease, dict):
        return False, "supervisor lease is not an object"
    for field in ROOT_IDENTITY_FIELDS:
        value = lease.get(field)
        if not isinstance(value, str) or not value:
            return False, f"supervisor lease {field} missing"
        if value != identity.get(field):
            return False, f"supervisor lease {field} mismatches root runtime"
    if "provider" in identity:
        value = lease.get("provider")
        if not isinstance(value, str) or not value:
            return False, "supervisor lease provider missing"
        if value != identity["provider"]:
            return False, "supervisor lease provider mismatches root runtime"
    if lease.get("root_agent", "root") != "root":
        return False, "supervisor lease selects a different root agent"
    if lease.get("runtime_agent_key", "root") != "root":
        return False, "supervisor lease selects a different agent"
    return True, "matched"


def _write_json_atomic(path: Path, value: dict) -> None:
    runtime.write_json_atomic(path, value)


def _status(goal: dict) -> str | None:
    value = goal.get("status") if isinstance(goal, dict) else None
    return value.strip().lower() if isinstance(value, str) else None


def root_snapshot(root: dict, previous: dict | None = None) -> dict:
    completions, cursor, rollout = runtime.rollout_task_snapshot(root, previous)
    return {"task_complete": completions, "rollout_cursor": cursor, "rollout": rollout}


def classify(goal: dict, snapshot: dict, queued: bool, now: float, idle_since: float | None,
             grace_period: float) -> tuple[str, str, float | None]:
    status = _status(goal)
    rollout = snapshot.get("rollout", {})
    active_turn = rollout.get("status") != "known" or bool(rollout.get("active_turns"))
    if status in BUDGET:
        return "ROOT_BUDGET_LIMITED", "platform budget or usage limit", idle_since
    if status in TERMINAL:
        return "ROOT_TERMINAL", f"Goal status={status}", idle_since
    if status in RESUMABLE:
        return "ROOT_PAUSED", f"Goal status={status}", idle_since
    if status not in ACTIVE:
        return "ROOT_TERMINAL", f"Goal status={status or 'unknown'}", idle_since
    if active_turn or queued:
        return "ROOT_RUNNING", "active turn or queued input", None
    if idle_since is None:
        idle_since = now
    if now - idle_since >= grace_period:
        return "ROOT_IDLE", "active Goal has no turn or queued input", idle_since
    return "ROOT_IDLE", "grace period", idle_since


def _wake_message(goal: dict, generation: int) -> str:
    return ROOT_LIVENESS_WAKE + json.dumps(
        {
            "reason": "root_goal_active_without_continuation",
            "goal_id": goal.get("goal_id"),
            "lease_generation": generation,
            "reentry": "liveness_only",
            "instruction": "Resume the root decision loop; do not infer a research task from this wake.",
        }, ensure_ascii=False, sort_keys=True
    )


def _budget_message(goal: dict) -> str:
    return ROOT_BUDGET_LIMITED + json.dumps(
        {"goal_id": goal.get("goal_id"), "status": goal.get("status"), "action": "notify_user_and_stop"},
        ensure_ascii=False, sort_keys=True,
    )


def _broker(args) -> AgentBroker | None:
    if not getattr(args, "broker_tasks_db", None) or not getattr(args, "broker_state_db", None):
        return None
    return AgentBroker(
        args.broker_tasks_db,
        args.broker_state_db,
        getattr(args, "broker_roles", "docs/AGENT_ROLES.yaml"),
        getattr(args, "broker_bindings", ".runtime/AGENT_BINDINGS.json"),
    )


def _resume_root_goal(root: dict, goal: dict, args):
    """Resume a paused or blocked root Goal through app-server and verify it."""

    command = getattr(args, "app_server_command", None)
    if not command and not (
        getattr(args, "codex_node", None) and getattr(args, "codex_js", None)
    ):
        raise RuntimeError(
            "root Goal recovery requires --codex-node/--codex-js or --app-server-command"
        )
    return runtime.app_server_resume_root_goal(
        root,
        goal,
        getattr(args, "codex_node", None),
        getattr(args, "codex_js", None),
        app_server_command=command,
        timeout=args.app_server_timeout,
    )


def check_once(args) -> dict:
    registry = runtime.read_json(args.registry)
    root = next(agent for agent in registry["agents"] if agent.get("agent_key") == "root")
    lease = load_lease(args.lease)
    identity, identity_reason = root_runtime_identity(args.registry, registry)
    lease_identity_ok, lease_identity_reason = lease_identity_matches(lease, identity)
    identity_ok = identity is not None and lease_identity_ok
    if identity_ok:
        identity_report = "matched"
    elif identity is None:
        identity_report = f"root runtime identity rejected: {identity_reason}"
    else:
        identity_report = f"root lease identity rejected: {lease_identity_reason}"
    state_path = Path(args.state)
    state = runtime.read_json(state_path) if state_path.is_file() else {"schema": SCHEMA}
    if state.get("schema") != SCHEMA:
        raise ValueError("incompatible root watchdog state schema")
    previous_snapshot = state.get("root_snapshot") if isinstance(state.get("root_snapshot"), dict) else None
    snapshot = root_snapshot(root, previous_snapshot)
    goal = runtime.goal_details(root)
    queued = runtime.pending_root_turn(root)
    now = time.time()
    state["root_snapshot"] = snapshot
    idle_since = state.get("idle_since")
    idle_since = float(idle_since) if isinstance(idle_since, (int, float)) else None
    status, reason, idle_since = classify(goal, snapshot, queued, now, idle_since, args.grace_period)
    state["idle_since"] = idle_since
    # Preserve the platform's exact Goal status even though paused and blocked
    # share the same bounded recovery path.
    state["observed_goal_status"] = _status(goal)
    state["last_status"] = status
    state["last_reason"] = reason
    state["lease_generation"] = lease.get("generation", 0)
    state["runtime_identity"] = {
        "status": "matched" if identity_ok else "rejected",
        "reason": identity_report,
    }
    state["updated_at"] = now

    # Identity is checked before constructing the broker.  A stale or
    # malformed binding/lease must be report-only: no CONTROL message,
    # budget notification, legacy queue, or app-server resume is allowed.
    if not identity_ok:
        state["last_reason"] = identity_report
        _write_json_atomic(state_path, state)
        print(identity_report, flush=True)
        return state

    broker = _broker(args)

    if broker is not None:
        supervisor = broker.status().get("supervisor") or {}
        state["broker_generation"] = supervisor.get("generation", 0)
        desired = str(supervisor.get("desired_state", "PAUSED")).upper()
        if desired != "RUNNING":
            state["last_reason"] = f"broker desired_state={desired}"
            _write_json_atomic(state_path, state)
            return state

    if status == "ROOT_BUDGET_LIMITED":
        if broker is not None:
            broker.control(action="BUDGET_LIMITED", target="root", reason="platform budget or usage limit")
        if not state.get("budget_notified"):
            print(_budget_message(goal), flush=True)
            state["budget_notified"] = True
        _write_json_atomic(state_path, state)
        return state

    if status == "ROOT_PAUSED":
        consumed = state.get("resume_consumed_goal_id")
        allowed = bool(lease.get("enabled") and lease.get("allow_root_resume"))
        if allowed and goal.get("goal_id") and consumed != goal.get("goal_id") and not args.dry_run:
            resumed = _resume_root_goal(root, goal, args)
            if broker is not None:
                broker.control(action="RESUME", target="root", reason="lease-authorized bounded root resume")
            resumed_goal = dict(goal)
            resumed_goal["status"] = resumed.get("status", "active")
            if not runtime.pending_root_turn(root):
                runtime.queue_root(root, _wake_message(resumed_goal, int(lease.get("generation", 0))), args.codex_node, args.codex_js)
            state["resume_consumed_goal_id"] = goal["goal_id"]
            state["last_status"] = "ROOT_RUNNING"
            state["last_reason"] = "lease-authorized Goal resume"
            print("RESUMED paused or blocked root Goal via app-server", flush=True)
        elif allowed and goal.get("goal_id") and consumed != goal.get("goal_id") and args.dry_run:
            print("DRY_RUN would resume root Goal via autonomy lease", flush=True)
        _write_json_atomic(state_path, state)
        return state

    if status == "ROOT_IDLE" and reason != "grace period" and not state.get("wake_sent"):
        if not lease.get("enabled"):
            state["last_reason"] = "autonomy lease disabled"
            _write_json_atomic(state_path, state)
            return state
        if not args.dry_run:
            if broker is not None:
                broker.control(action="WAKE", target="root", reason="root Goal active without continuation")
                state["wake_sent"] = True
                print("RECORDED ROOT_LIVENESS_WAKE in Agent Broker", flush=True)
                _write_json_atomic(state_path, state)
                return state
            if not runtime.pending_root_turn(root):
                runtime.queue_root(root, _wake_message(goal, int(lease.get("generation", 0))), args.codex_node, args.codex_js)
            state["wake_sent"] = True
            print("NOTIFIED ROOT_LIVENESS_WAKE", flush=True)
        else:
            print(_wake_message(goal, int(lease.get("generation", 0))), flush=True)
    if status == "ROOT_RUNNING":
        state["wake_sent"] = False
    _write_json_atomic(state_path, state)
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--lease", required=True)
    parser.add_argument("--codex-node")
    parser.add_argument("--codex-js")
    parser.add_argument("--grace-period", type=float, default=90.0)
    parser.add_argument("--app-server-timeout", type=float, default=30.0)
    parser.add_argument("--app-server-command", nargs="+")
    parser.add_argument("--broker-tasks-db")
    parser.add_argument("--broker-state-db")
    parser.add_argument("--broker-roles", default="docs/AGENT_ROLES.yaml")
    parser.add_argument("--broker-bindings", default=".runtime/AGENT_BINDINGS.json")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.grace_period <= 0:
        parser.error("--grace-period must be positive")
    if not (args.broker_tasks_db and args.broker_state_db) and not (args.codex_node and args.codex_js):
        parser.error("legacy mode requires --codex-node/--codex-js; broker mode requires both --broker-* databases")
    lock_path = Path(args.state + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another root watchdog owns this state file")
        while True:
            try:
                check_once(args)
            except Exception as error:  # pragma: no cover - daemon boundary
                print(f"WATCHDOG_ERROR {error}", file=sys.stderr, flush=True)
                if args.once:
                    return 1
            if args.once:
                return 0
            time.sleep(min(60.0, args.grace_period))


if __name__ == "__main__":
    raise SystemExit(main())
