#!/usr/bin/env python3
"""Observe fixed worker runtimes and send bounded events to ``/root``.

This daemon deliberately has no research policy and no Goal mutation path.  It
only reads worker snapshots and queues a ``POLL_EVENT`` when a registered
worker changes.  Root remains responsible for interpreting the event,
dispatching work, and integrating commits.

The older :mod:`agent_result_poller` remains available for historical state
files, but new deployments should use this module.  In particular, this
module never resumes a Goal and never emits a periodic decision wake.
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
    from scripts import agent_result_poller as legacy
except ModuleNotFoundError:  # pragma: no cover - direct CLI entry point
    import agent_result_poller as legacy


SCHEMA = "ref2dex.worker_event_poller.v1"
POLL_EVENT_PREFIX = "POLL_EVENT\n"
WORKER_ROLE_KEYS = frozenset({"agent_cm", "agent_rl", "agent_eval", "agent_infra"})


def _role_for(agent: dict) -> str | None:
    role = agent.get("role_key")
    if isinstance(role, str) and role in WORKER_ROLE_KEYS:
        return role
    # Compatibility for the v3 registry.  These mappings are intentionally
    # explicit; arbitrary agent creation must never become worker discovery.
    return {
        "agent_cm_temporal": "agent_cm",
        "agent_baseline": "agent_eval",
        "agent_c1_validation": "agent_eval",
        "agent_workflow": "agent_infra",
    }.get(agent.get("agent_key"))


def worker_agents(registry: dict, base_dir: str | Path | None = None) -> list[dict]:
    """Return registered worker runtimes, excluding root and daemon records."""

    # A local binding selects one replaceable conversation for each stable
    # role.  The registry still contains historical runtime records, but they
    # must not all be polled as if they were simultaneous fixed workers.
    selected: dict[str, str] = {}
    binding_path = registry.get("runtime_bindings", {}).get("path")
    if isinstance(binding_path, str):
        binding_path = Path(binding_path)
        if not binding_path.is_absolute() and base_dir is not None:
            binding_path = Path(base_dir) / binding_path
        try:
            bindings = json.loads(binding_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            bindings = None
        if isinstance(bindings, dict):
            for role, item in (bindings.get("bindings") or {}).items():
                runtime_key = item.get("runtime_agent_key") if isinstance(item, dict) else None
                if isinstance(runtime_key, str):
                    selected[role] = runtime_key

    result = []
    for agent in registry.get("agents", []):
        if not isinstance(agent, dict) or agent.get("agent_key") == "root":
            continue
        role = _role_for(agent)
        if role is None:
            continue
        if role in selected and selected[role] != agent.get("agent_key"):
            continue
        item = dict(agent)
        item["role_key"] = role
        result.append(item)
    return result


def _event_snapshot(snapshot: dict) -> dict:
    return legacy.event_snapshot(snapshot)


def _changes(before: dict, after: dict, worktree: str) -> list[str]:
    return legacy.changes(before, after, worktree)


def _observed_gpu_pids(event: dict) -> list[int]:
    return legacy._observed_gpu_pids(event)


def _message(events: list[dict]) -> str:
    if len(events) == 1:
        payload = events[0]
    else:
        payload = {"coalesced": True, "event_count": len(events), "events": events}
    return (
        POLL_EVENT_PREFIX
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        + "\nReview evidence and resource ownership before any mainline integration."
    )


def _merge_event(existing: dict, current: dict) -> dict:
    merged = dict(existing)
    merged["changed"] = list(dict.fromkeys(existing.get("changed", []) + current.get("changed", [])))
    merged["after"] = current.get("after", existing.get("after"))
    merged["observed_gpu_pids"] = sorted(
        set(existing.get("observed_gpu_pids", [])) | set(current.get("observed_gpu_pids", []))
    )
    if "task_complete" in merged["changed"] or "task_complete" in current.get("changed", []):
        ids = list(dict.fromkeys(existing.get("task_complete_added_ids", []) + current.get("task_complete_added_ids", [])))
        merged["task_complete_added_ids"] = ids
        merged["task_complete_added_count"] = len(ids)
    return merged


def _merge_events(previous: list[dict], current: list[dict]) -> list[dict]:
    merged: list[dict] = []
    positions: dict[str, int] = {}
    for event in [*(previous or []), *(current or [])]:
        key = event.get("agent_key")
        if key in positions:
            merged[positions[key]] = _merge_event(merged[positions[key]], event)
        else:
            positions[key] = len(merged)
            merged.append(event)
    return merged


def _root_poll_event_pending(root: dict) -> bool:
    return legacy.pending_root_poll_event(root)


def poll_once(args):
    registry = legacy.read_json(args.registry)
    root = next(agent for agent in registry["agents"] if agent.get("agent_key") == "root")
    state_path = Path(args.state)
    state = legacy.read_json(state_path) if state_path.is_file() else {"schema": SCHEMA, "agents": {}}
    if state.get("schema") != SCHEMA:
        raise ValueError("incompatible worker event poller state schema")

    all_pids = legacy.gpu_pids()
    known = state.setdefault("agents", {})
    current_snapshots = {}
    detected: list[dict] = []
    pending_snapshots: dict[str, dict] = {}
    for agent in worker_agents(registry, Path(args.registry).resolve().parent.parent):
        key = agent["agent_key"]
        current = legacy.snapshot(agent, all_pids, known.get(key))
        current["role_key"] = agent["role_key"]
        current_snapshots[key] = current
        previous = known.get(key)
        if previous is None:
            known[key] = current
            continue
        changed = _changes(previous, current, agent["worktree"])
        if not changed:
            known[key] = current
            continue
        event = {
            "agent_key": key,
            "role_key": agent["role_key"],
            "changed": changed,
            "before": _event_snapshot(previous),
            "after": _event_snapshot(current),
            "observed_gpu_pids": _observed_gpu_pids({"before": previous, "after": current}),
        }
        if "task_complete" in changed:
            old_ids = set(previous.get("task_complete", []))
            added = [item for item in current.get("task_complete", []) if item not in old_ids]
            event["task_complete_added_ids"] = added
            event["task_complete_added_count"] = len(added)
        detected.append(event)
        pending_snapshots[key] = current

    events = _merge_events(state.get("deferred_events", []), detected)
    if not events:
        if state.pop("deferred_events", None) is not None:
            legacy.write_json_atomic(state_path, state)
        else:
            legacy.write_json_atomic(state_path, state)
        return [], False

    state["deferred_events"] = events
    state["pending_snapshots"] = pending_snapshots
    legacy.write_json_atomic(state_path, state)
    if args.dry_run:
        print(_message(events), flush=True)
        return events, any(bool(item.get("gpu_pids")) for item in current_snapshots.values())

    if _root_poll_event_pending(root):
        print("DEFERRED root POLL_EVENT pending", flush=True)
        return events, any(bool(item.get("gpu_pids")) for item in current_snapshots.values())

    legacy.queue_root(root, _message(events), args.codex_node, args.codex_js)
    for key, current in pending_snapshots.items():
        known[key] = current
    state.pop("deferred_events", None)
    state.pop("pending_snapshots", None)
    legacy.write_json_atomic(state_path, state)
    print(f"NOTIFIED {len(events)} worker event(s)", flush=True)
    return events, any(bool(item.get("gpu_pids")) for item in current_snapshots.values())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--codex-node", required=True)
    parser.add_argument("--codex-js", required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.interval < 120:
        parser.error("interval must be at least 120 seconds")
    lock_path = Path(args.state + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another worker event poller owns this state file")
        while True:
            try:
                _, active_gpu = poll_once(args)
            except Exception as error:  # pragma: no cover - daemon boundary
                print(f"POLL_ERROR {error}", file=sys.stderr, flush=True)
                if args.once:
                    return 1
                active_gpu = False
            if args.once:
                return 0
            time.sleep(120 if active_gpu else args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
