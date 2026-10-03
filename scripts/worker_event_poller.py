#!/usr/bin/env python3
# DRAIN_ONLY: legacy workflow; new tasks use researchctl and docs/workflow/README.md.
"""Observe fixed worker runtimes and record bounded events in the Agent Broker.

This daemon deliberately has no research policy and no Goal mutation path.  It
only reads worker snapshots and writes a ``TASK_UPDATE`` when a registered
worker changes.  Root remains responsible for interpreting the event,
dispatching work, and integrating commits.  Without broker arguments, the
legacy queue path remains available for migration checks.

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
    from scripts.agent_broker import AgentBroker
except ModuleNotFoundError:  # pragma: no cover - direct CLI entry point
    import agent_result_poller as legacy
    from agent_broker import AgentBroker


SCHEMA = "ref2dex.worker_event_poller.v1"
POLL_EVENT_PREFIX = "POLL_EVENT\n"
WORKER_ROLE_KEYS = frozenset({"agent_cm", "agent_rl", "agent_eval", "agent_infra"})
RETIRING_LIFECYCLES = frozenset({"retiring", "retired", "unbound", "unbound_after_cleanup"})


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
    """Return only runtimes selected by a valid local fixed-pool binding.

    ``docs/AGENT_REGISTRY.json`` intentionally retains old runtime records.
    The machine-local binding is the source of truth for which replaceable
    conversation is live.  Missing, malformed, or mismatched binding data is
    treated as no worker rather than falling back to every historical record.
    """

    metadata = registry.get("runtime_bindings")
    binding_path = metadata.get("path") if isinstance(metadata, dict) else None
    if not isinstance(binding_path, str) or not binding_path:
        return []
    path = Path(binding_path)
    if not path.is_absolute():
        path = Path(base_dir or Path.cwd()) / path
    try:
        binding_doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    binding_map = binding_doc.get("bindings") if isinstance(binding_doc, dict) else None
    if not isinstance(binding_map, dict):
        return []

    # Index non-retiring registry records by key.  Retiring records can never
    # satisfy a live binding, even when an old binding still names them.
    candidates: dict[str, dict] = {}
    for agent in registry.get("agents", []):
        if not isinstance(agent, dict) or agent.get("agent_key") == "root":
            continue
        if str(agent.get("lifecycle", "")).strip().lower() in RETIRING_LIFECYCLES:
            continue
        key = agent.get("agent_key")
        if isinstance(key, str):
            candidates[key] = agent

    result = []
    used_runtime_keys: set[str] = set()
    for role in sorted(WORKER_ROLE_KEYS):
        binding = binding_map.get(role)
        if not isinstance(binding, dict) or binding.get("status") not in {"bound", "active"}:
            continue
        runtime_key = binding.get("runtime_agent_key")
        if not isinstance(runtime_key, str) or not runtime_key or runtime_key in used_runtime_keys:
            continue
        agent = candidates.get(runtime_key)
        if agent is None or _role_for(agent) != role:
            continue
        # When identity fields are present in the binding, every matching
        # registry value must agree.  A mismatch is skipped (fail closed) and
        # never silently replaced by another legacy runtime.
        mismatch = False
        for field in ("conversation_id", "codex_home", "worktree", "branch"):
            expected = binding.get(field)
            if expected is not None and expected != agent.get(field):
                mismatch = True
                break
        if mismatch:
            continue
        item = dict(agent)
        item["role_key"] = role
        result.append(item)
        used_runtime_keys.add(runtime_key)
    return result


def _event_snapshot(snapshot: dict) -> dict:
    return legacy.event_snapshot(snapshot)


def _changes(before: dict, after: dict, worktree: str) -> list[str]:
    return legacy.changes(before, after, worktree)


def _manifest_transition_key(before: dict, after: dict, worktree: str | None = None):
    """Return the canonical manifest target introduced by a transition."""

    return legacy.manifest_transition_key(before, after, worktree)


def _event_manifest_transition_key(event: dict):
    """Recover a canonical transition key from a queued event digest."""

    if not isinstance(event, dict) or "newest_manifest" not in event.get("changed", []):
        return None
    before = event.get("before") if isinstance(event.get("before"), dict) else {}
    after = event.get("after") if isinstance(event.get("after"), dict) else {}
    previous = legacy.manifest_identity(before.get("newest_manifest"))
    current = legacy.manifest_identity(after.get("newest_manifest"))
    return current if current is not None else previous


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


def _broker(args) -> AgentBroker | None:
    if not getattr(args, "broker_tasks_db", None) or not getattr(args, "broker_state_db", None):
        return None
    return AgentBroker(
        args.broker_tasks_db,
        args.broker_state_db,
        getattr(args, "broker_roles", "docs/AGENT_ROLES.yaml"),
        getattr(args, "broker_bindings", ".runtime/AGENT_BINDINGS.json"),
    )


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
    # Deferred events are still awaiting root acknowledgement.  Seed the
    # canonical set from them so an alias worktree cannot produce a second
    # notification for the same manifest while the first is pending.
    manifest_alerted = {
        key for key in (_event_manifest_transition_key(event)
                        for event in state.get("deferred_events", []))
        if key is not None
    }
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
        manifest_key = _manifest_transition_key(previous, current, agent["worktree"])
        if "newest_manifest" in changed and manifest_key in manifest_alerted:
            # The same canonical target may be visible through multiple
            # symlinked output roots or across a deferred poll cycle.  Keep
            # unrelated changes for this worker, but alert once per target.
            changed = [item for item in changed if item != "newest_manifest"]
        elif "newest_manifest" in changed:
            manifest_alerted.add(manifest_key)
        if not changed:
            # Even a deduplicated alias must advance its observation cursor;
            # otherwise every subsequent poll would rediscover the same edge.
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

    broker = _broker(args)
    if broker is not None:
        for event in events:
            # Legacy runtime records may still be selected by the binding
            # file; the broker identity is the stable role, never that old
            # conversation key.
            broker.observe(agent_key=event["role_key"], payload=event, task_id=None)
        for key, current in pending_snapshots.items():
            known[key] = current
        state.pop("deferred_events", None)
        state.pop("pending_snapshots", None)
        legacy.write_json_atomic(state_path, state)
        print(f"RECORDED {len(events)} worker event(s) in Agent Broker", flush=True)
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
    parser.add_argument("--codex-node")
    parser.add_argument("--codex-js")
    parser.add_argument("--broker-tasks-db")
    parser.add_argument("--broker-state-db")
    parser.add_argument("--broker-roles", default="docs/AGENT_ROLES.yaml")
    parser.add_argument("--broker-bindings", default=".runtime/AGENT_BINDINGS.json")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.interval < 120:
        parser.error("interval must be at least 120 seconds")
    if not (args.broker_tasks_db and args.broker_state_db) and not (args.codex_node and args.codex_js):
        parser.error("legacy mode requires --codex-node/--codex-js; broker mode requires both --broker-* databases")
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
