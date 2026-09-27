#!/usr/bin/env python3
"""Keep the root supervision turn live through the Agent Broker.

With broker databases configured, the watchdog writes only ``CONTROL``
messages and its own JSON state.  Legacy app-server wake/resume is retained
only when broker arguments are omitted.  It never reads experiment metrics,
starts a worker, or selects a research route.
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
    if status in TERMINAL or status == "blocked":
        return "ROOT_TERMINAL", f"Goal status={status}", idle_since
    if status == "paused":
        return "ROOT_PAUSED", "Goal is paused", idle_since
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


def check_once(args) -> dict:
    registry = runtime.read_json(args.registry)
    root = next(agent for agent in registry["agents"] if agent.get("agent_key") == "root")
    lease = load_lease(args.lease)
    broker = _broker(args)
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
    state["last_status"] = status
    state["last_reason"] = reason
    state["lease_generation"] = lease.get("generation", 0)
    state["updated_at"] = now

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
        if broker is not None and allowed and goal.get("goal_id") and consumed != goal.get("goal_id") and not args.dry_run:
            broker.control(action="RESUME", target="root", reason="lease-authorized bounded root resume")
            state["resume_consumed_goal_id"] = goal["goal_id"]
            state["last_status"] = "ROOT_RUNNING"
            state["last_reason"] = "broker-authorized Goal resume"
            print("RECORDED root Goal resume via Agent Broker", flush=True)
            _write_json_atomic(state_path, state)
            return state
        if allowed and goal.get("goal_id") and consumed != goal.get("goal_id") and not args.dry_run:
            resumed = runtime.app_server_resume_root_goal(
                root, goal, args.codex_node, args.codex_js,
                app_server_command=args.app_server_command,
                timeout=args.app_server_timeout,
            )
            state["resume_consumed_goal_id"] = goal["goal_id"]
            resumed_goal = dict(goal)
            resumed_goal["status"] = resumed.get("status", "active")
            if not runtime.pending_root_turn(root):
                runtime.queue_root(root, _wake_message(resumed_goal, int(lease.get("generation", 0))), args.codex_node, args.codex_js)
            state["last_status"] = "ROOT_RUNNING"
            state["last_reason"] = "lease-authorized Goal resume"
            print("RESUMED root Goal via autonomy lease", flush=True)
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
