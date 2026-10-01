#!/usr/bin/env python3
"""Validate the local fixed-role workflow before dispatching work.

The doctor is intentionally read-only.  It checks the tracked role contract,
local bindings, Broker database shape/state, active leases and the supervisor
lease.  A warning is useful for a deliberately unbound optional role; use
``--require-all-bound`` when a run needs every role available.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sqlite3
import sys
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError as error:  # pragma: no cover
    raise SystemExit(f"PyYAML is required: {error}")

try:  # Works both as ``python -m scripts...`` and a direct script path.
    from scripts.agent_broker import ADAPTERS, SCHEMA, load_bindings, load_roles
except ModuleNotFoundError:  # pragma: no cover - direct CLI entry point
    from agent_broker import ADAPTERS, SCHEMA, load_bindings, load_roles


def _result(level: str, check: str, detail: str) -> dict[str, str]:
    return {"level": level, "check": check, "detail": detail}


def inspect_workflow(
    *, roles_path: Path, registry_path: Path, bindings_path: Path,
    tasks_db: Path, state_db: Path, lease_path: Path,
    require_all_bound: bool = False, adapter_command: str | None = None,
    require_adapter: bool = False,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    roles = load_roles(roles_path)
    bindings = load_bindings(bindings_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_roles = {
        item.get("agent_key"): item
        for item in registry.get("pool", {}).get("roles", [])
        if isinstance(item, dict) and item.get("agent_key")
    }
    role_keys = set(roles)
    registry_keys = set(registry_roles)
    if role_keys != registry_keys:
        results.append(_result("ERROR", "role_keys", f"AGENT_ROLES differs from registry: {sorted(role_keys ^ registry_keys)}"))
    else:
        results.append(_result("OK", "role_keys", f"{len(role_keys)} fixed roles"))

    for key, role in sorted(roles.items()):
        registry_role = registry_roles.get(key, {})
        binding = bindings.get(key, {})
        if registry_role.get("branch") != role.get("branch"):
            results.append(_result("ERROR", f"branch:{key}", "registry branch differs from AGENT_ROLES"))
        elif binding.get("branch") not in (None, role.get("branch")):
            results.append(_result("ERROR", f"binding_branch:{key}", "binding branch differs from AGENT_ROLES"))
        elif key != "root" and binding.get("status") != "bound":
            level = "ERROR" if require_all_bound else "WARN"
            results.append(_result(level, f"binding:{key}", f"role is {binding.get('status', 'missing')}"))
        else:
            results.append(_result("OK", f"binding:{key}", "role and branch match"))
        provider = binding.get("provider")
        if binding.get("status") == "bound" and provider not in ADAPTERS:
            results.append(_result("ERROR", f"provider:{key}", f"unsupported provider {provider!r}"))

    if adapter_command is None:
        level = "ERROR" if require_adapter else "WARN"
        results.append(_result(level, "runtime_adapter", "no launcher command supplied"))
    else:
        try:
            argv = shlex.split(adapter_command)
            results.append(_result("OK" if argv else "ERROR", "runtime_adapter", "launcher command parsed" if argv else "empty launcher command"))
        except ValueError as error:
            results.append(_result("ERROR", "runtime_adapter", f"launcher command is invalid: {error}"))

    if not lease_path.is_file():
        results.append(_result("WARN", "supervisor_lease", "lease file is absent; automatic recovery is disabled"))
    else:
        lease = json.loads(lease_path.read_text(encoding="utf-8"))
        if lease.get("root_agent") != "root":
            results.append(_result("ERROR", "supervisor_lease", "lease does not select root"))
        else:
            results.append(_result("OK", "supervisor_lease", f"enabled={bool(lease.get('enabled'))}"))

    expected_task_tables = {"tasks", "messages", "handoffs"}
    expected_state_tables = {"runtime_state", "supervisor_state"}
    for path, expected, check in ((tasks_db, expected_task_tables, "tasks_db"), (state_db, expected_state_tables, "state_db")):
        if not path.is_file():
            results.append(_result("ERROR", check, "database is missing"))
            continue
        with sqlite3.connect(path) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            missing = expected - tables
            results.append(_result("ERROR" if missing else "OK", check, f"missing tables: {sorted(missing)}" if missing else "schema present"))

    if tasks_db.is_file():
        with sqlite3.connect(tasks_db) as db:
            active = db.execute(
                "SELECT target_agent, COUNT(*) FROM tasks WHERE status IN ('LEASED','RUNNING','HANDOFF_READY') GROUP BY target_agent HAVING COUNT(*) > 1"
            ).fetchall()
            expired = db.execute(
                "SELECT COUNT(*) FROM tasks WHERE status IN ('LEASED','RUNNING','HANDOFF_READY') AND lease_expires_at IS NOT NULL AND lease_expires_at <= strftime('%s','now')"
            ).fetchone()[0]
        results.append(_result("ERROR" if active else "OK", "single_active_task", f"duplicate active roles: {active}" if active else "at most one active task per role"))
        results.append(_result("ERROR" if expired else "OK", "lease_expiry", f"{expired} expired active leases"))

    if state_db.is_file():
        with sqlite3.connect(state_db) as db:
            row = db.execute("SELECT desired_state FROM supervisor_state WHERE singleton=1").fetchone()
        state = str(row[0]).upper() if row else "MISSING"
        results.append(_result("ERROR" if state == "MISSING" else "OK", "supervisor_state", state))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles", type=Path, default=Path("docs/AGENT_ROLES.yaml"))
    parser.add_argument("--registry", type=Path, default=Path("docs/AGENT_REGISTRY.json"))
    parser.add_argument("--bindings", type=Path, default=Path(".runtime/AGENT_BINDINGS.json"))
    parser.add_argument("--tasks-db", type=Path, default=Path(".runtime/tasks.sqlite"))
    parser.add_argument("--state-db", type=Path, default=Path(".runtime/AGENT_STATE.sqlite"))
    parser.add_argument("--lease", type=Path, default=Path(".runtime/SUPERVISOR_LEASE.json"))
    parser.add_argument("--adapter-command")
    parser.add_argument("--require-all-bound", action="store_true")
    parser.add_argument("--require-adapter", action="store_true")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        results = inspect_workflow(
            roles_path=args.roles, registry_path=args.registry, bindings_path=args.bindings,
            tasks_db=args.tasks_db, state_db=args.state_db, lease_path=args.lease,
            require_all_bound=args.require_all_bound, adapter_command=args.adapter_command,
            require_adapter=args.require_adapter,
        )
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"workflow doctor failed: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for item in results:
            print(f"{item['level']:5} {item['check']}: {item['detail']}")
    bad = {"ERROR"} | ({"WARN"} if args.strict else set())
    return 1 if any(item["level"] in bad for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
