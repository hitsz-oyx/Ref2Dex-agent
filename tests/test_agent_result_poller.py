from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from scripts import agent_result_poller as poller


def _child(tmp_path: Path, key: str) -> tuple[dict, Path]:
    thread = f"{key}-thread"
    home = tmp_path / f"{key}-home"
    rollout_dir = home / "sessions" / "2026" / "09"
    rollout_dir.mkdir(parents=True)
    rollout = rollout_dir / f"rollout-2026-09-26T00-00-00-{thread}.jsonl"
    rollout.touch()
    worktree = tmp_path / f"{key}-worktree"
    worktree.mkdir()
    return (
        {
            "agent_key": key,
            "conversation_id": thread,
            "codex_home": str(home),
            "rollout_locator": f"sessions/**/rollout-*-{thread}.jsonl",
            "worktree": str(worktree),
            "goal_db": str(home / "goals_1.sqlite"),
        },
        rollout,
    )


def _args(registry: Path, state: Path, *, dry_run: bool = False):
    return SimpleNamespace(
        registry=str(registry),
        state=str(state),
        dry_run=dry_run,
        codex_node="node",
        codex_js="codex.js",
    )


def _record(turn_id: str, *, root_turn_id: str | None = None) -> str:
    payload = {"type": "task_complete", "turn_id": turn_id}
    if root_turn_id is not None:
        payload["root_turn_id"] = root_turn_id
    return json.dumps({"type": "event_msg", "payload": payload}) + "\n"


def _setup(monkeypatch):
    monkeypatch.setattr(poller, "git_head", lambda worktree: "same-head")
    monkeypatch.setattr(
        poller,
        "goal_state",
        lambda agent: {"goal_id": "same-goal", "status": "active"},
    )
    monkeypatch.setattr(poller, "gpu_pids", lambda: [])
    monkeypatch.setattr(poller, "owned_gpu_pids", lambda worktree, pids: [])


def test_initial_snapshot_is_silent_then_own_completion_emits_once(tmp_path, monkeypatch):
    _setup(monkeypatch)
    child, rollout = _child(tmp_path, "child")
    rollout.write_text(
        "".join(_record(f"old-{index}") for index in range(100)), encoding="utf-8"
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "agents": [
                    {"agent_key": "root", "conversation_id": "root-thread"},
                    child,
                ]
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "state.json"
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    args = _args(registry, state)

    events, _ = poller.poll_once(args)
    assert events == []
    assert queued == []

    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(_record("new"))
        handle.write(_record("new"))
    events, _ = poller.poll_once(args)
    assert len(events) == 1
    assert events[0]["changed"] == ["task_complete"]
    assert events[0]["before"]["task_complete"]["count"] == 100
    assert events[0]["after"]["task_complete"]["count"] == 101
    assert events[0]["task_complete_added_ids"] == ["turn:new"]
    assert queued and len(queued) == 1
    assert len(queued[0]) < 1200

    events, _ = poller.poll_once(args)
    assert events == []
    assert len(queued) == 1


def test_delegated_completion_in_registered_rollout_is_ignored(tmp_path, monkeypatch):
    _setup(monkeypatch)
    child, rollout = _child(tmp_path, "child")
    rollout.write_text(_record("nested", root_turn_id="parent"), encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "agents": [
                    {"agent_key": "root", "conversation_id": "root-thread"},
                    child,
                ]
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "state.json"
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    args = _args(registry, state)

    assert poller.poll_once(args)[0] == []
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(_record("own", root_turn_id="own"))
    events, _ = poller.poll_once(args)
    assert len(events) == 1
    assert events[0]["changed"] == ["task_complete"]
    assert len(queued) == 1


def test_shared_manifest_symlink_emits_one_alert(tmp_path, monkeypatch):
    _setup(monkeypatch)
    child_a, _ = _child(tmp_path, "child_a")
    child_b, _ = _child(tmp_path, "child_b")
    shared = tmp_path / "shared-output"
    shared.mkdir()
    manifest = shared / "run_manifest.json"
    manifest.write_text("{\"version\": 1}\n", encoding="utf-8")

    target_a = tmp_path / "output-target-a"
    target_a.mkdir()
    (target_a / "run_manifest.json").symlink_to(manifest)
    (Path(child_a["worktree"]) / "outputs").symlink_to(target_a, target_is_directory=True)

    target_b = tmp_path / "output-target-b"
    (target_b / "nested").mkdir(parents=True)
    (target_b / "nested" / "run_manifest.json").symlink_to(manifest)
    (Path(child_b["worktree"]) / "outputs").symlink_to(target_b, target_is_directory=True)

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "agents": [
                    {"agent_key": "root", "conversation_id": "root-thread"},
                    child_a,
                    child_b,
                ]
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "state.json"
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    args = _args(registry, state)

    assert poller.poll_once(args)[0] == []
    # Different symlink aliases are the same manifest identity, not a change.
    assert poller.poll_once(args)[0] == []
    old_mtime = manifest.stat().st_mtime_ns
    manifest.write_text("{\"version\": 2}\n", encoding="utf-8")
    os.utime(manifest, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))

    events, _ = poller.poll_once(args)
    assert len(events) == 1
    assert events[0]["changed"] == ["newest_manifest"]
    assert len(queued) == 1
    assert poller.poll_once(args)[0] == []
