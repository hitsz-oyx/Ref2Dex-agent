#!/usr/bin/env python3
"""Watch registered child agents and notify root only when their state changes."""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SCHEMA = "ref2dex.agent_result_poller.v1"


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
        candidate = {"path": str(path.relative_to(worktree)), "mtime_ns": modified}
        if newest is None or (modified, candidate["path"]) > (newest["mtime_ns"], newest["path"]):
            newest = candidate
    return newest


def snapshot(agent, all_gpu_pids):
    return {
        "head": git_head(agent["worktree"]),
        "goal": goal_state(agent),
        "gpu_pids": owned_gpu_pids(agent["worktree"], all_gpu_pids),
        "newest_manifest": newest_manifest(agent["worktree"]),
    }


def changes(before, after):
    return [key for key in ("head", "goal", "gpu_pids", "newest_manifest")
            if before.get(key) != after.get(key)]


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
    for agent in registry["agents"]:
        key = agent["agent_key"]
        if key in {"root", "agent_poller"}:
            continue
        current = snapshot(agent, pids)
        active_gpu = active_gpu or bool(current["gpu_pids"])
        previous = known.get(key)
        if previous is None:
            known[key] = current
            continue
        changed = changes(previous, current)
        if not changed:
            continue
        event = {"agent_key": key, "changed": changed, "before": previous,
                 "after": current}
        events.append(event)
        message = "POLL_EVENT\n" + json.dumps(event, ensure_ascii=False, sort_keys=True)
        message += "\nReview evidence and resource ownership before any mainline integration."
        if args.dry_run:
            print(message, flush=True)
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
