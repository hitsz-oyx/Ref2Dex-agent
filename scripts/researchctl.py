#!/usr/bin/env python3
"""Small local control plane for the Ref2Dex autonomy lease."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:  # Works both as ``python -m scripts...`` and a direct script path.
    from scripts import agent_result_poller as runtime
    from scripts.agent_broker import AgentBroker
    from scripts.root_watchdog import LEASE_SCHEMA, load_lease
except ModuleNotFoundError:  # pragma: no cover - direct CLI entry point
    import agent_result_poller as runtime
    from agent_broker import AgentBroker
    from root_watchdog import LEASE_SCHEMA, load_lease


def write_lease(path: Path, *, enabled: bool, mode: str) -> dict:
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
    runtime.write_json_atomic(path, value)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("supervisor", choices=["supervisor"])
    parser.add_argument("action", choices=["pause", "resume", "status"])
    parser.add_argument("--lease", type=Path, default=Path(".runtime/SUPERVISOR_LEASE.json"))
    parser.add_argument("--broker-tasks-db", type=Path)
    parser.add_argument("--broker-state-db", type=Path)
    parser.add_argument("--broker-roles", type=Path, default=Path("docs/AGENT_ROLES.yaml"))
    parser.add_argument("--broker-bindings", type=Path, default=Path(".runtime/AGENT_BINDINGS.json"))
    args = parser.parse_args(argv)
    if args.action == "status":
        value = {"lease": load_lease(args.lease)}
        if args.broker_state_db:
            broker = AgentBroker(args.broker_tasks_db or ".runtime/tasks.sqlite", args.broker_state_db,
                                 args.broker_roles, args.broker_bindings)
            value["broker"] = broker.status().get("supervisor")
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    value = write_lease(args.lease, enabled=args.action == "resume", mode="autonomous" if args.action == "resume" else "manual")
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
