#!/usr/bin/env python3
"""Watch registered child agents and coalesce state or turn completion changes."""

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SCHEMA = "ref2dex.agent_result_poller.v1"
TASK_COMPLETE = "task_complete"
POLL_EVENT_PREFIX = "POLL_EVENT\n"
ROOT_DECISION_WAKE_PREFIX = "ROOT_DECISION_WAKE\n"

# Only these terminal/waiting states are safe evidence that an execution
# owner is idle.  Unknown values are deliberately not treated as idle: a
# missed status transition must never manufacture a supervisory wake.
IDLE_GOAL_STATUSES = frozenset({
    "none", "paused", "blocked", "usage_limited", "complete", "completed",
    "terminal", "failed", "stopped", "terminated", "waiting", "idle",
})
ACTIVE_ROOT_GOAL_STATUSES = frozenset({"active", "running"})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent),
                                     prefix=path.name + ".", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(str(temp_path), str(path))


def git_head(worktree):
    result = subprocess.run(["git", "-C", worktree, "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def goal_state(agent):
    db = Path(agent["goal_db"])
    if not db.is_file():
        return {"status": "UNKNOWN", "goal_id": None}
    connection = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT goal_id, status FROM thread_goals WHERE thread_id=? "
            "ORDER BY updated_at_ms DESC LIMIT 1", (agent["conversation_id"],)
        ).fetchone()
    finally:
        connection.close()
    return {"goal_id": row[0], "status": row[1]} if row else {"goal_id": None, "status": "NONE"}


def gpu_pids():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        return []
    return [int(line.strip()) for line in result.stdout.splitlines()
            if line.strip().isdigit()]


def owned_gpu_pids(worktree, all_pids):
    root = str(Path(worktree).resolve()) + os.sep
    owned = []
    for pid in all_pids:
        try:
            cwd = os.readlink("/proc/{}/cwd".format(pid))
        except OSError:
            continue
        if cwd == root[:-1] or cwd.startswith(root):
            owned.append(pid)
    return sorted(owned)


def rollout_paths(agent):
    """Return the registered thread's rollout files, deduplicated by target."""

    home = Path(agent["codex_home"])
    locator = agent.get("rollout_locator")
    if not isinstance(locator, str) or not locator:
        return []
    try:
        candidates = list(home.glob(locator))
    except (NotImplementedError, OSError, RuntimeError, ValueError):
        return []
    paths = {}
    for candidate in candidates:
        try:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            continue
        paths[str(resolved)] = resolved
    return [paths[key] for key in sorted(paths)]


def _task_complete_payload(record):
    return _task_event_payload(record, TASK_COMPLETE)


def _task_event_payload(record, event_type):
    if not isinstance(record, dict):
        return None
    if record.get("type") == event_type:
        return record
    payload = record.get("payload")
    if isinstance(payload, dict) and payload.get("type") == event_type:
        return payload
    return None


def _is_own_task_complete(payload):
    """Ignore delegated child turns recorded inside a registered thread's rollout."""

    turn_id = payload.get("turn_id")
    root_turn_id = payload.get("root_turn_id")
    if root_turn_id and root_turn_id != turn_id:
        return False
    return bool(turn_id) or not root_turn_id


def _is_own_task_event(payload):
    """Apply the same root-turn filter to task_started and task_complete."""

    return _is_own_task_complete(payload)


def _task_turn_id(payload):
    turn_id = payload.get("turn_id")
    if isinstance(turn_id, str) and turn_id:
        return "turn:" + turn_id
    return None


def _task_complete_id(payload):
    turn_id = payload.get("turn_id")
    if isinstance(turn_id, str) and turn_id:
        return "turn:" + turn_id
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "event:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_rollout_lines(rollout, offset, seen, active_turns):
    """Read appended records and return completion/active/cursor evidence."""

    additions = []
    cursor = offset
    partial = False
    try:
        with rollout.open("rb") as handle:
            handle.seek(offset)
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line:
                    cursor = handle.tell()
                    break
                if not line.endswith(b"\n"):
                    # Keep a partial record for the next poll.  It may be a
                    # task_started event, so callers must conservatively treat
                    # the rollout as unknown until the line is complete.
                    cursor = start
                    partial = True
                    break
                cursor = handle.tell()
                try:
                    record = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    continue
                started = _task_event_payload(record, "task_started")
                if started is not None and _is_own_task_event(started):
                    identifier = _task_turn_id(started)
                    if identifier is None:
                        partial = True
                    else:
                        active_turns.add(identifier)
                completed = _task_complete_payload(record)
                if completed is not None and _is_own_task_event(completed):
                    identifier = _task_turn_id(completed)
                    if identifier is not None:
                        active_turns.discard(identifier)
                    completion_id = _task_complete_id(completed)
                    if completion_id not in seen:
                        seen.add(completion_id)
                        additions.append(completion_id)
                aborted = _task_event_payload(record, "turn_aborted")
                if aborted is not None and _is_own_task_event(aborted):
                    # Aborted turns are terminal for active-turn tracking but
                    # are not completed results and must not create IDs.
                    identifier = _task_turn_id(aborted)
                    if identifier is not None:
                        active_turns.discard(identifier)
    except OSError:
        return [], offset, active_turns, False, True
    return additions, cursor, active_turns, not partial, False


def rollout_task_snapshot(agent, previous=None):
    """Read completion IDs and an append cursor for a registered thread.

    Rollouts are append-only JSONL files.  A prior cursor makes normal polls
    read only the newly appended bytes; a missing or invalid cursor causes one
    catch-up scan, which remains quiet when it only confirms the old snapshot.
    """

    previous = previous if isinstance(previous, dict) else {}
    completions = list(previous.get("task_complete", []))
    seen = set(completions)
    old_cursors = previous.get("rollout_cursor")
    old_cursors = old_cursors if isinstance(old_cursors, dict) else {}
    old_rollout = previous.get("rollout")
    old_rollout = old_rollout if isinstance(old_rollout, dict) else {}
    old_active = old_rollout.get("active_turns", [])
    old_active = set(item for item in old_active if isinstance(item, str))
    cursors = {}
    paths = rollout_paths(agent)
    # A pre-CPU-detector state may already have a byte cursor but no active
    # turn evidence.  Re-scan it once so an unmatched historical start cannot
    # be mistaken for an idle NONE Goal after an upgrade.
    full_scan = not old_cursors or "active_turns" not in old_rollout
    path_stats = {}
    for rollout in paths:
        key = str(rollout)
        try:
            stat = rollout.stat()
        except OSError:
            continue
        path_stats[key] = stat
        old = old_cursors.get(key)
        if not isinstance(old, dict):
            full_scan = True
            continue
        same_file = old.get("inode") == stat.st_ino and old.get("device") == stat.st_dev
        old_offset = old.get("offset")
        if not same_file or not isinstance(old_offset, int) or not 0 <= old_offset <= stat.st_size:
            full_scan = True
    active_turns = set() if full_scan else old_active
    known = bool(paths) and not full_scan
    unknown = not bool(paths)
    for rollout in paths:
        key = str(rollout)
        stat = path_stats.get(key)
        if stat is None:
            unknown = True
            continue
        old = old_cursors.get(key)
        offset = 0
        if not full_scan and isinstance(old, dict):
            same_file = old.get("inode") == stat.st_ino and old.get("device") == stat.st_dev
            old_offset = old.get("offset")
            if same_file and isinstance(old_offset, int) and 0 <= old_offset <= stat.st_size:
                offset = old_offset
        additions, offset, active_turns, readable, failed = _read_rollout_lines(
            rollout, offset, seen, active_turns
        )
        completions.extend(additions)
        cursors[key] = {"device": stat.st_dev, "inode": stat.st_ino, "offset": offset}
        if failed or not readable:
            unknown = True
        else:
            known = True
    if not paths:
        # A transiently unavailable session must not look like completed turns
        # disappearing from the thread.
        cursors = dict(old_cursors)
    rollout_state = {
        "status": "known" if known and not unknown else "unknown",
        "active_turns": sorted(active_turns),
    }
    return completions, cursors, rollout_state


def rollout_task_completions(agent):
    """Compatibility wrapper returning only completion IDs."""

    return rollout_task_snapshot(agent)[0]


def _canonical_path(path):
    try:
        return str(Path(path).resolve())
    except (OSError, RuntimeError):
        return str(Path(path).absolute())


def newest_manifest(worktree):
    output_root = Path(worktree) / "outputs"
    if not output_root.is_dir():
        return None
    newest = None
    for directory, _, files in os.walk(str(output_root)):
        if "run_manifest.json" not in files:
            continue
        path = Path(directory) / "run_manifest.json"
        try:
            modified = path.stat().st_mtime_ns
        except OSError:
            continue
        candidate = {
            "path": str(path.relative_to(worktree)),
            "canonical_path": _canonical_path(path),
            "mtime_ns": modified,
        }
        if newest is None or (modified, candidate["path"]) > (newest["mtime_ns"], newest["path"]):
            newest = candidate
    return newest


def manifest_identity(manifest, worktree=None):
    if not isinstance(manifest, dict):
        return None
    canonical = manifest.get("canonical_path")
    if not canonical:
        relative = manifest.get("path")
        if not relative:
            return None
        canonical = Path(worktree, relative) if worktree else relative
        canonical = _canonical_path(canonical)
    return (str(canonical), manifest.get("mtime_ns"))


def snapshot(agent, all_gpu_pids, previous=None):
    completions, rollout_cursor, rollout = rollout_task_snapshot(agent, previous)
    return {
        "head": git_head(agent["worktree"]),
        "goal": goal_state(agent),
        "gpu_pids": owned_gpu_pids(agent["worktree"], all_gpu_pids),
        "newest_manifest": newest_manifest(agent["worktree"]),
        "task_complete": completions,
        "rollout_cursor": rollout_cursor,
        "rollout": rollout,
    }


def changes(before, after, worktree=None):
    changed = [key for key in ("head", "goal", "gpu_pids")
               if before.get(key) != after.get(key)]
    if manifest_identity(before.get("newest_manifest"), worktree) != manifest_identity(
        after.get("newest_manifest"), worktree
    ):
        changed.append("newest_manifest")
    # A state file written by the previous poller version has no completion
    # cursor. Treat its first read as the baseline so historical turns do not
    # wake root during a watcher upgrade.
    if "task_complete" in before and before.get("task_complete") != after.get("task_complete"):
        changed.append("task_complete")
    return changed


def manifest_transition_key(before, after, worktree=None):
    previous = manifest_identity(before.get("newest_manifest"), worktree)
    current = manifest_identity(after.get("newest_manifest"), worktree)
    if previous == current:
        return None
    # Aliased worktrees can have different old relative paths (or one can have
    # no visible manifest yet) while observing the same canonical target now.
    # Key a transition by the target being introduced/updated; this makes the
    # per-poll dedup independent of the alias-specific history.
    return current if current is not None else previous


def event_snapshot(snapshot):
    """Keep queue messages bounded while retaining full IDs in the state file."""

    summary = dict(snapshot)
    summary.pop("rollout_cursor", None)
    completions = summary.get("task_complete")
    if isinstance(completions, list):
        summary["task_complete"] = {
            "count": len(completions),
            "latest": completions[-1] if completions else None,
        }
    return summary


def _observed_gpu_pids(*events):
    observed = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        values = event.get("observed_gpu_pids", [])
        if isinstance(values, list):
            observed.update(pid for pid in values if isinstance(pid, int))
        for side in (event.get("before"), event.get("after")):
            if isinstance(side, dict) and isinstance(side.get("gpu_pids"), list):
                observed.update(pid for pid in side["gpu_pids"] if isinstance(pid, int))
    return sorted(observed)


def _queue_item_text(payload_json):
    """Extract user text from a Codex queue payload without trusting its shape."""

    try:
        payload = json.loads(payload_json)
    except (TypeError, ValueError):
        return ""
    user_input = payload.get("UserInput") if isinstance(payload, dict) else None
    content = user_input.get("content") if isinstance(user_input, dict) else None
    if not isinstance(content, list):
        return ""
    texts = []
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            texts.append(item["text"])
    return "\n".join(texts)


def pending_root_poll_event(root_agent):
    """Return whether an unconsumed POLL_EVENT is already queued for root.

    Queue inspection errors are raised so the caller never acknowledges a
    change while the deduplication source is unavailable.
    """

    database = root_agent.get("queue_db")
    thread_id = root_agent.get("conversation_id")
    if not database or not thread_id:
        # Synthetic registries and old handoffs may omit the queue path.  There
        # is no queue to gate in that case, so retain the historical behavior.
        return False
    database = Path(database)
    if not database.is_file():
        return False
    connection = None
    try:
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        rows = connection.execute(
            "SELECT payload_json FROM queued_items WHERE thread_id=?",
            (thread_id,),
        )
        return any(_queue_item_text(payload).startswith(POLL_EVENT_PREFIX)
                   for (payload,) in rows)
    except (OSError, sqlite3.DatabaseError) as error:
        raise RuntimeError("cannot inspect root queue {}: {}".format(database, error)) from error
    finally:
        if connection is not None:
            connection.close()


def pending_root_turn(root_agent):
    """Return whether any queued input is already waiting for root.

    A decision wake is only safe when root has no other queued turn.  Missing
    queue metadata is retained as the historical synthetic-test behavior
    (there is no durable queue to gate); an actual queue inspection failure is
    raised so the caller can retry without acknowledging state.
    """

    database = root_agent.get("queue_db")
    thread_id = root_agent.get("conversation_id")
    if not database or not thread_id:
        return False
    database = Path(database)
    if not database.is_file():
        return False
    connection = None
    try:
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        row = connection.execute(
            "SELECT 1 FROM queued_items WHERE thread_id=? LIMIT 1", (thread_id,)
        ).fetchone()
        return row is not None
    except (OSError, sqlite3.DatabaseError) as error:
        raise RuntimeError("cannot inspect root queue {}: {}".format(database, error)) from error
    finally:
        if connection is not None:
            connection.close()


def _normal_goal_status(goal):
    status = goal.get("status") if isinstance(goal, dict) else None
    return status.strip().lower() if isinstance(status, str) else None


def execution_idle_snapshot(children, snapshots):
    """Classify registered execution owners without inferring unknown state.

    Returns ``(all_idle, details)``.  A GPU PID or an active/unknown Goal
    status keeps the result non-idle.  The details are bounded to the
    registered child set and are persisted for an auditable transition edge.
    """

    details = {}
    if not children:
        return False, details
    all_idle = True
    for agent in children:
        key = agent["agent_key"]
        current = snapshots.get(key, {})
        status = _normal_goal_status(current.get("goal"))
        gpu = list(current.get("gpu_pids") or [])
        rollout = current.get("rollout")
        rollout_status = rollout.get("status") if isinstance(rollout, dict) else "unknown"
        active_turns = list(rollout.get("active_turns", [])) if isinstance(rollout, dict) else []
        if gpu:
            idle = False
            reason = "gpu"
        elif rollout_status != "known":
            idle = False
            reason = "rollout_unknown"
        elif active_turns:
            idle = False
            reason = "cpu_turn"
        elif status in IDLE_GOAL_STATUSES:
            idle = True
            reason = "goal"
        else:
            idle = False
            reason = "unknown_goal" if status is None or status == "unknown" else "active_goal"
        details[key] = {
            "status": status,
            "gpu_pids": gpu,
            "rollout_status": rollout_status,
            "active_turns": active_turns,
            "idle": idle,
            "reason": reason,
        }
        all_idle = all_idle and idle
    return all_idle, details


def _root_goal_is_active(root_goal):
    return _normal_goal_status(root_goal) in ACTIVE_ROOT_GOAL_STATUSES


def decision_wake_message(root_goal, execution_details):
    """Build one bounded, non-resuming supervisory re-entry notification."""

    payload = {
        "reason": "execution_agents_became_all_idle",
        "root_goal": root_goal,
        "execution_agents": execution_details,
        "reentry": "event_only",
        "goal_resume": "user_or_system_controlled",
    }
    return ROOT_DECISION_WAKE_PREFIX + json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    ) + "\nChoose the next authorized decision; this event does not resume Goal."


def poll_event_message(events):
    """Build one bounded notification for this poll cycle.

    A single event keeps the legacy wire shape.  Multiple child changes are
    wrapped in one ordered digest so root receives one wake-up while retaining
    each child and every newly completed turn ID.
    """

    if len(events) == 1:
        payload = events[0]
    else:
        payload = {
            "coalesced": True,
            "event_count": len(events),
            "events": events,
        }
    return POLL_EVENT_PREFIX + json.dumps(payload, ensure_ascii=False, sort_keys=True) + \
        "\nReview evidence and resource ownership before any mainline integration."


def _merge_event(existing, current):
    """Merge two observations for one child while retaining all new IDs."""

    merged = dict(existing)
    merged["changed"] = list(dict.fromkeys(
        list(existing.get("changed", [])) + list(current.get("changed", []))
    ))
    merged["after"] = current.get("after", existing.get("after"))
    merged["observed_gpu_pids"] = _observed_gpu_pids(existing, current)
    if "task_complete" in merged["changed"] or "task_complete" in current.get("changed", []):
        ids = list(existing.get("task_complete_added_ids", []))
        ids.extend(current.get("task_complete_added_ids", []))
        ids = list(dict.fromkeys(ids))
        merged["task_complete_added_ids"] = ids
        merged["task_complete_added_count"] = len(ids)
    return merged


def merge_events(previous, current):
    """Coalesce deferred and newly observed child events by agent key."""

    merged = []
    positions = {}
    for event in list(previous or []) + list(current or []):
        key = event.get("agent_key")
        if key in positions:
            index = positions[key]
            merged[index] = _merge_event(merged[index], event)
        else:
            positions[key] = len(merged)
            merged.append(event)
    return merged


def queue_root(root_agent, message, node_bin, codex_js):
    environment = os.environ.copy()
    environment["CODEX_HOME"] = root_agent["codex_home"]
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        environment[name] = "http://127.0.0.1:7897"
    result = subprocess.run(
        [node_bin, codex_js, "queue", "--thread", root_agent["conversation_id"],
         "--message", message],
        env=environment, capture_output=True, text=True, timeout=60, check=False,
    )
    if result.returncode:
        raise RuntimeError("root queue failed: {}".format((result.stderr or result.stdout).strip()))


def poll_once(args):
    registry = read_json(args.registry)
    root = next(agent for agent in registry["agents"] if agent["agent_key"] == "root")
    state_path = Path(args.state)
    state = read_json(state_path) if state_path.is_file() else {"schema": SCHEMA, "agents": {}}
    if state.get("schema") != SCHEMA:
        raise ValueError("incompatible poller state schema")
    known = state["agents"]
    pids = gpu_pids()
    events = []
    active_gpu = False
    children = []
    for agent in registry["agents"]:
        key = agent["agent_key"]
        if key in {"root", "agent_poller"}:
            continue
        children.append(agent)
    current_snapshots = {}
    for agent in children:
        key = agent["agent_key"]
        current = snapshot(agent, pids, known.get(key))
        current_snapshots[key] = current
        active_gpu = active_gpu or bool(current["gpu_pids"])
    root_goal = goal_state(root)
    all_execution_idle, execution_details = execution_idle_snapshot(
        children, current_snapshots
    )
    previous_supervision = state.get("supervision")
    previous_supervision = previous_supervision if isinstance(previous_supervision, dict) else {}
    previous_all_idle = previous_supervision.get("all_execution_idle")
    supervision = {
        "all_execution_idle": all_execution_idle,
        "root_goal_status": root_goal.get("status"),
        "execution_agents": execution_details,
        "wake_pending": bool(previous_supervision.get("wake_pending")),
        "wake_sent": bool(previous_supervision.get("wake_sent")),
    }
    if not all_execution_idle:
        supervision["wake_pending"] = False
        supervision["wake_sent"] = False
    elif previous_all_idle is None:
        # Establish a quiet baseline when upgrading an old state file.
        supervision["wake_pending"] = False
        supervision["wake_sent"] = False
    elif previous_all_idle is False:
        supervision["wake_pending"] = _root_goal_is_active(root_goal)
        supervision["wake_sent"] = False
    elif not _root_goal_is_active(root_goal):
        # A paused, usage-limited, or unknown root is never auto-woken.  The
        # all-idle edge is consumed so a later manual resume does not replay it.
        supervision["wake_pending"] = False
        supervision["wake_sent"] = False

    # A shared outputs symlink can expose the same manifest to several
    # worktrees. Alert once for a canonical manifest transition, while still
    # advancing every agent's cursor so the duplicate cannot reappear later.
    manifest_alerted = set()
    pending_updates = {}
    detected_events = []
    for agent in children:
        key = agent["agent_key"]
        current = current_snapshots[key]
        previous = known.get(key)
        if previous is None:
            known[key] = current
            continue
        changed = changes(previous, current, agent["worktree"])
        if not changed:
            # Also migrate old snapshots to the completion/canonical-manifest
            # representation without producing an event.
            known[key] = current
            continue
        manifest_key = manifest_transition_key(previous, current, agent["worktree"])
        if "newest_manifest" in changed and manifest_key in manifest_alerted:
            changed = [item for item in changed if item != "newest_manifest"]
        elif "newest_manifest" in changed:
            manifest_alerted.add(manifest_key)
        if not changed:
            known[key] = current
            continue
        event = {"agent_key": key, "changed": changed,
                 "before": event_snapshot(previous), "after": event_snapshot(current)}
        event["observed_gpu_pids"] = _observed_gpu_pids(event)
        if "task_complete" in changed:
            previous_ids = set(previous.get("task_complete", []))
            added = [identifier for identifier in current["task_complete"]
                     if identifier not in previous_ids]
            event["task_complete_added_count"] = len(added)
            event["task_complete_added_ids"] = added
        detected_events.append(event)
        pending_updates[key] = current

    deferred_events = state.get("deferred_events", [])
    if not isinstance(deferred_events, list):
        deferred_events = []
    events = merge_events(deferred_events, detected_events)
    if events:
        message = poll_event_message(events)
        if args.dry_run:
            print(message, flush=True)
            for key, current in pending_updates.items():
                known[key] = current
        else:
            pending_snapshots = state.get("pending_snapshots", {})
            pending_snapshots = pending_snapshots if isinstance(pending_snapshots, dict) else {}
            pending_snapshots.update(pending_updates)
            # Make the event and the exact observed snapshot durable before
            # inspecting or writing the root queue.  This closes the loss
            # window where a transient GPU PID disappears after queue failure.
            state["deferred_events"] = events
            state["pending_snapshots"] = pending_snapshots
            state["supervision"] = supervision
            write_json_atomic(state_path, state)
            if pending_root_poll_event(root):
                # The previous notification is already durable in root's
                # queue.  Keep the digest and observed snapshot until it is
                # consumed, including any transient GPU PID.
                print("DEFERRED root POLL_EVENT pending", flush=True)
            else:
                queue_root(root, message, args.codex_node, args.codex_js)
                for key, current in pending_snapshots.items():
                    known[key] = current
                state.pop("deferred_events", None)
                state.pop("pending_snapshots", None)
                print("NOTIFIED {} event(s)".format(len(events)), flush=True)
    elif deferred_events:
        # A malformed or stale deferred list should not persist forever once
        # all of its events have been acknowledged by a successful queue.
        state.pop("deferred_events", None)
    elif supervision.get("wake_pending") and not args.dry_run:
        # A child-idle transition may have coincided with an earlier queued
        # POLL_EVENT.  Keep the one-shot wake pending until root has no queued
        # turn, then deliver it exactly once.  Queue failures persist the
        # pending bit and are retried by the next sparse poll.
        try:
            if pending_root_turn(root):
                print("DEFERRED root ROOT_DECISION_WAKE pending", flush=True)
            elif _root_goal_is_active(root_goal):
                queue_root(
                    root,
                    decision_wake_message(root_goal, execution_details),
                    args.codex_node,
                    args.codex_js,
                )
                supervision["wake_pending"] = False
                supervision["wake_sent"] = True
                print("NOTIFIED ROOT_DECISION_WAKE", flush=True)
        except Exception:
            # Do not lose the transition when queue or read-only inspection
            # fails.  The caller still reports POLL_ERROR, but the persisted
            # pending marker makes the next invocation retry safely.
            state["supervision"] = supervision
            if not args.dry_run:
                write_json_atomic(state_path, state)
            raise
    if not args.dry_run:
        state["supervision"] = supervision
    if not args.dry_run:
        write_json_atomic(state_path, state)
    return events, active_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--codex-node", required=True)
    parser.add_argument("--codex-js", required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.interval < 120:
        parser.error("interval must be at least 120 seconds")
    lock_path = Path(args.state + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another poller already owns this state file")
    while True:
        active_gpu = False
        try:
            _, active_gpu = poll_once(args)
        except Exception as error:
            print("POLL_ERROR {}".format(error), file=sys.stderr, flush=True)
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(120 if active_gpu else args.interval)


if __name__ == "__main__":
    sys.exit(main())
