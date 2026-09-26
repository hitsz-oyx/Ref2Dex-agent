#!/usr/bin/env python3
"""Watch registered child agents and notify root on state or turn completion changes."""

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
    if not isinstance(record, dict):
        return None
    if record.get("type") == TASK_COMPLETE:
        return record
    payload = record.get("payload")
    if isinstance(payload, dict) and payload.get("type") == TASK_COMPLETE:
        return payload
    return None


def _is_own_task_complete(payload):
    """Ignore delegated child turns recorded inside a registered thread's rollout."""

    turn_id = payload.get("turn_id")
    root_turn_id = payload.get("root_turn_id")
    if root_turn_id and root_turn_id != turn_id:
        return False
    return bool(turn_id) or not root_turn_id


def _task_complete_id(payload):
    turn_id = payload.get("turn_id")
    if isinstance(turn_id, str) and turn_id:
        return "turn:" + turn_id
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "event:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def rollout_task_completions(agent):
    """Read stable completion IDs from the registered agent's own rollout."""

    completions = []
    seen = set()
    for rollout in rollout_paths(agent):
        try:
            handle = rollout.open("r", encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, ValueError):
                    # A live JSONL writer may leave a partial final record.
                    continue
                payload = _task_complete_payload(record)
                if payload is None or not _is_own_task_complete(payload):
                    continue
                identifier = _task_complete_id(payload)
                if identifier not in seen:
                    seen.add(identifier)
                    completions.append(identifier)
    return completions


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


def snapshot(agent, all_gpu_pids):
    return {
        "head": git_head(agent["worktree"]),
        "goal": goal_state(agent),
        "gpu_pids": owned_gpu_pids(agent["worktree"], all_gpu_pids),
        "newest_manifest": newest_manifest(agent["worktree"]),
        "task_complete": rollout_task_completions(agent),
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
        current = snapshot(agent, pids)
        current_snapshots[key] = current
        active_gpu = active_gpu or bool(current["gpu_pids"])

    # A shared outputs symlink can expose the same manifest to several
    # worktrees. Alert once for a canonical manifest transition, while still
    # advancing every agent's cursor so the duplicate cannot reappear later.
    manifest_alerted = set()
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
        event = {"agent_key": key, "changed": changed, "before": previous,
                 "after": current}
        events.append(event)
        message = "POLL_EVENT\n" + json.dumps(event, ensure_ascii=False, sort_keys=True)
        message += "\nReview evidence and resource ownership before any mainline integration."
        if args.dry_run:
            print(message, flush=True)
            known[key] = current
            continue
        queue_root(root, message, args.codex_node, args.codex_js)
        known[key] = current
        write_json_atomic(state_path, state)
        print("NOTIFIED {} {}".format(key, ",".join(changed)), flush=True)
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
