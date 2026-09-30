#!/usr/bin/env python3
"""Drain-only compatibility control for the retired Goal/lease workflow."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:  # Works both as ``python -m scripts...`` and a direct script path.
    from scripts import agent_result_poller as runtime
    from scripts.agent_broker import AgentBroker
    from scripts.root_watchdog import LEASE_SCHEMA, load_lease, root_runtime_identity
except ModuleNotFoundError:  # pragma: no cover - direct CLI entry point
    import agent_result_poller as runtime
    from agent_broker import AgentBroker
    from root_watchdog import LEASE_SCHEMA, load_lease, root_runtime_identity


LEASE_IDENTITY_FIELDS = ("conversation_id", "codex_home", "worktree", "branch", "provider")


def write_lease(path: Path, *, enabled: bool, mode: str,
                registry: Path | None = None) -> dict:
    current = load_lease(path)
    value = {
        "schema": LEASE_SCHEMA,
        "enabled": enabled,
        "mode": mode,
        "root_agent": "root",
        "allow_root_resume": enabled,
        "generation": int(current.get("generation", 0) or 0) + 1,
        "updated_at": time.time(),
    }
    # Preserve an existing identity while migrating or pausing a lease. This
    # prevents a pause/resume cycle from deleting watchdog-required fields.
    for field in LEASE_IDENTITY_FIELDS:
        if isinstance(current.get(field), str) and current[field]:
            value[field] = current[field]
    if registry is not None:
        identity, reason = root_runtime_identity(registry)
        if identity is None:
            if enabled:
                raise ValueError(f"cannot enable supervisor lease without root runtime identity: {reason}")
        else:
            value.update(identity)
    # Local migration fixtures may omit a registry; root_watchdog continues to
    # fail closed until a complete identity is supplied.
    runtime.write_json_atomic(path, value)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("supervisor", choices=["supervisor"])
    parser.add_argument("action", choices=["pause", "resume", "status"])
    parser.add_argument("--lease", type=Path, default=Path(".runtime/SUPERVISOR_LEASE.json"))
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--broker-tasks-db", type=Path)
    parser.add_argument("--broker-state-db", type=Path)
    parser.add_argument("--broker-roles", type=Path, default=Path("docs/AGENT_ROLES.yaml"))
    parser.add_argument("--broker-bindings", type=Path, default=Path(".runtime/AGENT_BINDINGS.json"))
    args = parser.parse_args(argv)
    if args.registry is None:
        candidate = Path("docs/AGENT_REGISTRY.json")
        args.registry = candidate if candidate.is_file() else None
    if bool(args.broker_tasks_db) != bool(args.broker_state_db):
        parser.error("broker control plane requires both --broker-tasks-db and --broker-state-db")
    if args.action == "status":
        value = {"lease": load_lease(args.lease)}
        if args.broker_state_db:
            broker = AgentBroker(args.broker_tasks_db or ".runtime/tasks.sqlite", args.broker_state_db,
                                 args.broker_roles, args.broker_bindings)
            value["broker"] = broker.status().get("supervisor")
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    value = write_lease(
        args.lease,
        enabled=args.action == "resume",
        mode="autonomous" if args.action == "resume" else "manual",
        registry=args.registry,
    )
    broker_value = None
    if args.broker_state_db:
        broker = AgentBroker(args.broker_tasks_db or ".runtime/tasks.sqlite", args.broker_state_db,
                             args.broker_roles, args.broker_bindings)
        broker_value = broker.control(
            action="RESUME" if args.action == "resume" else "PAUSE",
            target="root",
            reason=f"researchctl {args.action}",
        )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    if broker_value is not None:
        print(json.dumps(broker_value, ensure_ascii=False, indent=2, sort_keys=True))
    if args.action == "pause":
        print("Lease disabled. Pause the root Goal through the Codex UI/API if an immediate stop is required.")
    else:
        print("Lease enabled. The root watchdog may resume a paused root Goal on its next bounded check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
