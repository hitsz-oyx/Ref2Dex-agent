#!/usr/bin/env python3
"""Consume Broker tasks and hand them to an explicit runtime launcher.

The Broker deliberately does not know how to start Codex, NewAPI, or another
provider.  This small adapter closes that boundary without hiding it: one
adapter process owns one fixed worker role, claims a lease, sends a canonical
TASK_DISPATCH JSON document to the configured launcher on stdin, renews the
lease while the launcher is alive, and requires the launcher/worker to write a
TASK_HANDOFF back through the Broker.

The launcher is intentionally supplied by the caller.  The adapter never
inherits a root account or guesses a provider command from a display name.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:  # Works both as ``python -m scripts...`` and a direct script path.
    from scripts.agent_broker import AgentBroker, BrokerError, adapter_for, load_bindings
except ModuleNotFoundError:  # pragma: no cover - direct CLI entry point
    from agent_broker import AgentBroker, BrokerError, adapter_for, load_bindings


ACTIVE_TASK_STATES = frozenset({"LEASED", "RUNNING", "HANDOFF_READY"})


def _task_from_status(broker: AgentBroker, task_id: str) -> dict[str, Any] | None:
    for task in broker.status().get("tasks", []):
        if task.get("task_id") == task_id:
            return task
    return None


def _request_payload(task: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    request = adapter_for(binding).dispatch(task, binding)
    return {
        "type": "TASK_DISPATCH",
        "task_id": task["task_id"],
        "target": task["target_agent"],
        "objective": task["objective"],
        "context": task["context"],
        "constraints": task["constraints"],
        "done_when": task["done_when"],
        "provider": request.provider,
        "runtime_agent_key": request.runtime_agent_key,
        "provider_request": request.payload,
        "runtime_binding": {
            key: binding.get(key)
            for key in ("profile", "conversation_id", "codex_home", "worktree", "branch")
            if binding.get(key) is not None
        },
    }


def _write_request(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{payload['task_id']}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def _fail_if_active(broker: AgentBroker, task: dict[str, Any], reason: str) -> None:
    current = _task_from_status(broker, task["task_id"])
    if not current or current.get("status") not in ACTIVE_TASK_STATES:
        return
    try:
        broker.handoff(
            task_id=task["task_id"],
            agent_key=task["target_agent"],
            status="FAILED",
            result={"runtime_error": reason},
            evidence=["runtime_adapter"],
            recommended_next_action="inspect runtime launcher and retry the task explicitly",
            lease_token=task["lease_token"],
        )
    except BrokerError as error:
        print(f"RUNTIME_ADAPTER_HANDOFF_ERROR {error}", file=sys.stderr, flush=True)


def run_one(broker: AgentBroker, binding: dict[str, Any], task: dict[str, Any],
            command: str, request_dir: Path | None, lease_seconds: float) -> dict[str, Any]:
    payload = _request_payload(task, binding)
    _write_request(request_dir, payload)
    argv = shlex.split(command)
    if not argv:
        raise ValueError("runtime command is empty")
    environment = os.environ.copy()
    environment.update({
        "REF2DEX_TASK_ID": str(task["task_id"]),
        "REF2DEX_AGENT_KEY": str(task["target_agent"]),
        "REF2DEX_LEASE_TOKEN": str(task["lease_token"]),
        "REF2DEX_TASKS_DB": str(broker.tasks_db),
        "REF2DEX_STATE_DB": str(broker.state_db),
        "REF2DEX_ROLES": str(broker.roles_path),
        "REF2DEX_BINDINGS": str(broker.bindings_path),
    })
    if isinstance(binding.get("codex_home"), str) and binding["codex_home"]:
        environment["CODEX_HOME"] = binding["codex_home"]
    worktree = binding.get("worktree")
    if not isinstance(worktree, str) or not worktree:
        raise ValueError("runtime binding has no worktree")
    process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        text=True,
        env=environment,
        cwd=worktree,
    )
    assert process.stdin is not None
    process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    process.stdin.close()
    heartbeat = max(0.2, min(30.0, lease_seconds / 3.0))
    next_heartbeat = time.monotonic() + heartbeat
    while process.poll() is None:
        time.sleep(min(0.2, max(0.0, next_heartbeat - time.monotonic())))
        if process.poll() is not None:
            break
        if time.monotonic() < next_heartbeat:
            continue
        try:
            broker.renew(
                task_id=task["task_id"],
                agent_key=task["target_agent"],
                lease_token=task["lease_token"],
                lease_seconds=lease_seconds,
            )
            next_heartbeat = time.monotonic() + heartbeat
        except BrokerError as error:
            process.terminate()
            process.wait(timeout=5)
            _fail_if_active(broker, task, f"lease renewal failed: {error}")
            return {"task_id": task["task_id"], "status": "FAILED", "reason": str(error)}
    returncode = process.wait()
    current = _task_from_status(broker, task["task_id"])
    if returncode != 0:
        _fail_if_active(broker, task, f"runtime launcher exited with code {returncode}")
    elif current and current.get("status") in ACTIVE_TASK_STATES:
        _fail_if_active(broker, task, "runtime launcher exited without TASK_HANDOFF")
    current = _task_from_status(broker, task["task_id"])
    return {
        "task_id": task["task_id"],
        "status": current.get("status") if current else "UNKNOWN",
        "runtime_exit_code": returncode,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--command", required=True, help="launcher command; receives TASK_DISPATCH JSON on stdin")
    parser.add_argument("--tasks-db", default=".runtime/tasks.sqlite")
    parser.add_argument("--state-db", default=".runtime/AGENT_STATE.sqlite")
    parser.add_argument("--roles", default="docs/AGENT_ROLES.yaml")
    parser.add_argument("--bindings", default=".runtime/AGENT_BINDINGS.json")
    parser.add_argument("--request-dir", type=Path)
    parser.add_argument("--lease-seconds", type=float, default=900.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.lease_seconds <= 0 or args.poll_interval <= 0:
        parser.error("lease and poll intervals must be positive")
    bindings = load_bindings(args.bindings)
    binding = bindings.get(args.agent)
    if not isinstance(binding, dict) or binding.get("status") != "bound":
        parser.error(f"agent has no bound runtime: {args.agent}")
    broker = AgentBroker(args.tasks_db, args.state_db, args.roles, args.bindings)
    while True:
        try:
            task = broker.claim(args.agent, lease_seconds=args.lease_seconds)
        except BrokerError as error:
            if args.once:
                print(json.dumps({"status": "NOT_RUN", "reason": str(error)}), flush=True)
                return 0
            time.sleep(args.poll_interval)
            continue
        if task is None:
            if args.once:
                print(json.dumps({"status": "IDLE"}), flush=True)
                return 0
            time.sleep(args.poll_interval)
            continue
        try:
            result = run_one(broker, binding, task, args.command, args.request_dir, args.lease_seconds)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            _fail_if_active(broker, task, f"could not start runtime launcher: {error}")
            result = {"task_id": task["task_id"], "status": "FAILED", "reason": str(error)}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if args.once:
            return 0 if result.get("status") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
