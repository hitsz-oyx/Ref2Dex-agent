from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        root_goal_resume_once=False,
        app_server_command=None,
        app_server_timeout=3.0,
    )


def _record(turn_id: str, *, root_turn_id: str | None = None) -> str:
    payload = {"type": "task_complete", "turn_id": turn_id}
    if root_turn_id is not None:
        payload["root_turn_id"] = root_turn_id
    return json.dumps({"type": "event_msg", "payload": payload}) + "\n"


def _started_record(turn_id: str, *, root_turn_id: str | None = None) -> str:
    payload = {"type": "task_started", "turn_id": turn_id}
    if root_turn_id is not None:
        payload["root_turn_id"] = root_turn_id
    return json.dumps({"type": "event_msg", "payload": payload}) + "\n"


def _aborted_record(turn_id: str) -> str:
    return json.dumps(
        {"type": "event_msg", "payload": {"type": "turn_aborted", "turn_id": turn_id}}
    ) + "\n"


def _setup(monkeypatch):
    monkeypatch.setattr(poller, "git_head", lambda worktree: "same-head")
    monkeypatch.setattr(
        poller,
        "goal_state",
        lambda agent: {"goal_id": "same-goal", "status": "active"},
    )
    monkeypatch.setattr(poller, "gpu_pids", lambda: [])
    monkeypatch.setattr(poller, "owned_gpu_pids", lambda worktree, pids: [])


def _registry(tmp_path: Path, *children: dict, root_queue: Path | None = None) -> Path:
    root = {"agent_key": "root", "conversation_id": "root-thread"}
    if root_queue is not None:
        root["queue_db"] = str(root_queue)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"agents": [root, *children]}), encoding="utf-8"
    )
    return registry


def _queued_items_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE queued_items ("
        "id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, payload_json TEXT NOT NULL, "
        "queue_order INTEGER NOT NULL, created_at_ms INTEGER NOT NULL, "
        "updated_at_ms INTEGER NOT NULL)"
    )
    connection.commit()
    return connection


def _pending_poll_payload() -> str:
    return json.dumps(
        {
            "UserInput": {
                "content": [{"text": "POLL_EVENT\n{\"agent_key\":\"old\"}"}]
            }
        }
    )


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


def test_completion_backlog_is_one_coalesced_notification(tmp_path, monkeypatch):
    _setup(monkeypatch)
    child_a, rollout_a = _child(tmp_path, "child_a")
    child_b, rollout_b = _child(tmp_path, "child_b")
    rollout_a.write_text(_record("old-a") + _record("new-a"), encoding="utf-8")
    rollout_b.write_text(_record("old-b") + _record("new-b"), encoding="utf-8")
    registry = _registry(tmp_path, child_a, child_b)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "schema": poller.SCHEMA,
                "agents": {
                    child_a["agent_key"]: {
                        "head": "same-head", "goal": {"goal_id": "same-goal", "status": "active"},
                        "gpu_pids": [], "newest_manifest": None,
                        "task_complete": ["turn:old-a"],
                    },
                    child_b["agent_key"]: {
                        "head": "same-head", "goal": {"goal_id": "same-goal", "status": "active"},
                        "gpu_pids": [], "newest_manifest": None,
                        "task_complete": ["turn:old-b"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )

    events, _ = poller.poll_once(_args(registry, state))
    assert len(events) == 2
    assert len(queued) == 1
    message_payload = json.loads(queued[0].split("\n", 1)[1].split("\nReview", 1)[0])
    assert message_payload["coalesced"] is True
    assert message_payload["event_count"] == 2
    assert {
        event["task_complete_added_ids"][0] for event in message_payload["events"]
    } == {"turn:new-a", "turn:new-b"}
    assert poller.poll_once(_args(registry, state))[0] == []
    assert len(queued) == 1


def test_pending_root_event_defers_and_then_catches_up_after_restart(tmp_path, monkeypatch):
    _setup(monkeypatch)
    child, rollout = _child(tmp_path, "child")
    rollout.write_text(_record("old"), encoding="utf-8")
    queue_db = tmp_path / "root-queue.sqlite"
    connection = _queued_items_db(queue_db)
    connection.close()
    registry = _registry(tmp_path, child, root_queue=queue_db)
    state = tmp_path / "state.json"
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    args = _args(registry, state)

    assert poller.poll_once(args)[0] == []
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(_record("new"))

    connection = sqlite3.connect(queue_db)
    connection.execute(
        "INSERT INTO queued_items VALUES (?, ?, ?, ?, ?, ?)",
        ("pending", "root-thread", _pending_poll_payload(), 0, 1, 1),
    )
    connection.commit()
    connection.close()

    # A fresh process with the persisted cursor must still defer while the
    # earlier root notification is queued.
    events, _ = poller.poll_once(_args(registry, state))
    assert len(events) == 1
    assert queued == []
    deferred_state = json.loads(state.read_text(encoding="utf-8"))
    assert deferred_state["agents"][child["agent_key"]]["task_complete"] == ["turn:old"]
    assert deferred_state["deferred_events"][0]["task_complete_added_ids"] == ["turn:new"]

    connection = sqlite3.connect(queue_db)
    connection.execute("DELETE FROM queued_items")
    connection.commit()
    connection.close()

    events, _ = poller.poll_once(_args(registry, state))
    assert len(events) == 1
    assert len(queued) == 1
    assert "turn:new" in queued[0]

    # A completion after catch-up is a distinct notification and is not
    # replayed with the already acknowledged backlog.
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(_record("newer"))
    events, _ = poller.poll_once(_args(registry, state))
    assert len(events) == 1
    assert len(queued) == 2
    assert "turn:newer" in queued[1]
    payload = json.loads(queued[1].split("\n", 1)[1].split("\nReview", 1)[0])
    assert payload["task_complete_added_ids"] == ["turn:newer"]
    assert poller.poll_once(args)[0] == []


def test_queue_failure_keeps_unacknowledged_completion_for_retry(tmp_path, monkeypatch):
    _setup(monkeypatch)
    child, rollout = _child(tmp_path, "child")
    rollout.write_text(_record("old"), encoding="utf-8")
    registry = _registry(tmp_path, child)
    state = tmp_path / "state.json"
    args = _args(registry, state)
    assert poller.poll_once(args)[0] == []
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(_record("new"))

    def fail_queue(root, message, node, codex):
        raise RuntimeError("synthetic queue outage")

    monkeypatch.setattr(poller, "queue_root", fail_queue)
    try:
        poller.poll_once(args)
    except RuntimeError as error:
        assert "synthetic queue outage" in str(error)
    else:
        raise AssertionError("queue failure must propagate without acknowledging state")
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["agents"][child["agent_key"]]["task_complete"] == ["turn:old"]

    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    assert len(poller.poll_once(args)[0]) == 1
    assert len(queued) == 1 and "turn:new" in queued[0]


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


def _supervision_setup(monkeypatch, statuses, root_status="active", root_goal_id=None):
    monkeypatch.setattr(poller, "git_head", lambda worktree: "same-head")
    monkeypatch.setattr(
        poller,
        "goal_state",
        lambda agent: {
            "goal_id": None if agent["agent_key"] == "root" else "child-goal",
            "status": root_status if agent["agent_key"] == "root"
            else statuses.get(agent["agent_key"], "active"),
        },
    )
    monkeypatch.setattr(
        poller,
        "goal_details",
        lambda agent: {
            "goal_id": root_goal_id if agent["agent_key"] == "root" else "child-goal",
            "status": root_status if agent["agent_key"] == "root"
            else statuses.get(agent["agent_key"], "active"),
            "objective": "root objective" if agent["agent_key"] == "root" else None,
            "token_budget": 7 if agent["agent_key"] == "root" else None,
        },
    )
    monkeypatch.setattr(poller, "gpu_pids", lambda: [])
    monkeypatch.setattr(poller, "owned_gpu_pids", lambda worktree, pids: [])


def test_all_idle_transition_queues_one_decision_wake(tmp_path, monkeypatch):
    statuses = {"child": "active"}
    _supervision_setup(monkeypatch, statuses)
    child, _ = _child(tmp_path, "child")
    registry = _registry(tmp_path, child)
    state = tmp_path / "state.json"
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    args = _args(registry, state)

    # First observation is a quiet non-idle baseline.
    assert poller.poll_once(args)[0] == []
    statuses["child"] = "paused"
    events, _ = poller.poll_once(args)
    assert events and events[0]["changed"] == ["goal"]
    assert len(queued) == 1  # the ordinary state-change POLL_EVENT
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["supervision"]["all_execution_idle"] is True
    assert persisted["supervision"]["wake_pending"] is True

    # Once the ordinary event has been handled, the next sparse poll emits
    # exactly one decision wake; unchanged idle polls stay quiet.
    assert poller.poll_once(args)[0] == []
    assert len(queued) == 2
    assert queued[1].startswith(poller.ROOT_DECISION_WAKE_PREFIX)
    assert poller.poll_once(args)[0] == []
    assert len(queued) == 2


def test_active_execution_owner_suppresses_decision_wake(tmp_path, monkeypatch):
    statuses = {"child": "active"}
    _supervision_setup(monkeypatch, statuses)
    child, _ = _child(tmp_path, "child")
    registry = _registry(tmp_path, child)
    state = tmp_path / "state.json"
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    args = _args(registry, state)
    assert poller.poll_once(args)[0] == []
    assert poller.poll_once(args)[0] == []
    assert queued == []


@pytest.mark.parametrize("root_status", ["paused", "UNKNOWN"])
def test_paused_or_unknown_root_never_auto_wakes(tmp_path, monkeypatch, root_status):
    statuses = {"child": "active"}
    _supervision_setup(monkeypatch, statuses, root_status=root_status)
    child, _ = _child(tmp_path, "child")
    registry = _registry(tmp_path, child)
    state = tmp_path / "state.json"
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    args = _args(registry, state)
    assert poller.poll_once(args)[0] == []
    statuses["child"] = "paused"
    assert poller.poll_once(args)[0]
    assert len(queued) == 1  # state-change event only
    assert poller.poll_once(args)[0] == []
    assert len(queued) == 1


@pytest.mark.parametrize("root_status", ["paused", "blocked"])
def test_opt_in_resume_uses_app_server_once_on_idle_edge(tmp_path, monkeypatch, root_status):
    statuses = {"child": "active"}
    _supervision_setup(monkeypatch, statuses, root_status=root_status, root_goal_id="goal-1")
    child, _ = _child(tmp_path, "child")
    registry = _registry(tmp_path, child)
    state = tmp_path / "state.json"
    queued = []
    resumed = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )

    def resume(root, goal, node, codex, app_server_command=None, timeout=30.0):
        resumed.append((root["conversation_id"], goal["goal_id"], goal["status"]))
        return {"threadId": root["conversation_id"], "status": "active"}

    monkeypatch.setattr(poller, "app_server_resume_root_goal", resume)
    args = _args(registry, state)
    args.root_goal_resume_once = True

    assert poller.poll_once(args)[0] == []
    statuses["child"] = "paused"
    assert poller.poll_once(args)[0]
    assert len(resumed) == 0
    assert len(queued) == 1  # the ordinary child state-change event

    assert poller.poll_once(args)[0] == []
    assert resumed == [("root-thread", "goal-1", root_status)]
    assert len(queued) == 2
    assert queued[-1].startswith(poller.ROOT_DECISION_WAKE_PREFIX)
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["supervision"]["root_goal_resume_consumed_goal_id"] == "goal-1"

    # The Goal row is deliberately still reported as paused.  The consumed
    # exact ID prevents a second automatic resume on unchanged idle polls.
    assert poller.poll_once(args)[0] == []
    assert len(resumed) == 1
    assert len(queued) == 2


def test_reblocked_after_overload_keeps_exact_goal_status_and_consumed_guard(
    tmp_path, monkeypatch,
):
    """Poller recovery remains one-shot when the same Goal blocks again."""
    statuses = {"child": "active"}
    root_status = {"value": "blocked"}
    monkeypatch.setattr(poller, "git_head", lambda worktree: "same-head")
    monkeypatch.setattr(
        poller,
        "goal_state",
        lambda agent: {
            "goal_id": None if agent["agent_key"] == "root" else "child-goal",
            "status": root_status["value"] if agent["agent_key"] == "root"
            else statuses.get(agent["agent_key"], "active"),
        },
    )
    monkeypatch.setattr(
        poller,
        "goal_details",
        lambda agent: {
            "goal_id": "goal-cycle" if agent["agent_key"] == "root" else "child-goal",
            "status": root_status["value"] if agent["agent_key"] == "root"
            else statuses.get(agent["agent_key"], "active"),
            "objective": "root objective" if agent["agent_key"] == "root" else None,
            "token_budget": 7 if agent["agent_key"] == "root" else None,
        },
    )
    monkeypatch.setattr(poller, "gpu_pids", lambda: [])
    monkeypatch.setattr(poller, "owned_gpu_pids", lambda worktree, pids: [])
    child, _ = _child(tmp_path, "child")
    registry = _registry(tmp_path, child)
    state = tmp_path / "state.json"
    queued = []
    resumed = []
    monkeypatch.setattr(poller, "queue_root", lambda *args: queued.append(args[1]))
    monkeypatch.setattr(
        poller,
        "app_server_resume_root_goal",
        lambda root, goal, node, codex, **kwargs: (
            resumed.append(goal["status"]) or {"threadId": "root-thread", "status": "active"}
        ),
    )
    args = _args(registry, state)
    args.root_goal_resume_once = True

    # Establish a non-idle baseline, then trigger the one-shot recovery edge.
    assert poller.poll_once(args)[0] == []
    statuses["child"] = "paused"
    assert poller.poll_once(args)[0]
    assert poller.poll_once(args)[0] == []
    assert resumed == ["blocked"]
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["supervision"]["root_goal_status"] == "active"
    assert persisted["supervision"]["root_goal_resume_consumed_goal_id"] == "goal-cycle"
    assert len(queued) == 2

    # The app-server overload leaves the same Goal blocked again.  It remains
    # classified as blocked, with no duplicate resume or wake notification.
    root_status["value"] = "blocked"
    assert poller.poll_once(args)[0] == []
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["supervision"]["root_goal_status"] == "blocked"
    assert persisted["supervision"]["root_goal_resume_consumed_goal_id"] == "goal-cycle"
    assert resumed == ["blocked"]
    assert len(queued) == 2


def test_static_migrated_idle_paused_goal_resumes_without_new_edge(tmp_path, monkeypatch):
    statuses = {"child": "paused"}
    _supervision_setup(monkeypatch, statuses, root_status="paused", root_goal_id="goal-static")
    child, _ = _child(tmp_path, "child")
    registry = _registry(tmp_path, child)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "schema": poller.SCHEMA,
        "agents": {},
        "supervision": {
            "all_execution_idle": True,
            "wake_pending": False,
            "wake_sent": False,
        },
    }), encoding="utf-8")
    resumed = []
    queued = []
    monkeypatch.setattr(poller, "queue_root", lambda *args: queued.append(args[1]))
    monkeypatch.setattr(
        poller,
        "app_server_resume_root_goal",
        lambda root, goal, node, codex, **kwargs: (
            resumed.append(goal["goal_id"]) or {"threadId": "root-thread", "status": "active"}
        ),
    )
    args = _args(registry, state)
    args.root_goal_resume_once = True
    assert poller.poll_once(args)[0] == []
    assert resumed == ["goal-static"]
    assert queued and queued[-1].startswith(poller.ROOT_DECISION_WAKE_PREFIX)


@pytest.mark.parametrize("goal_status", ["paused", "blocked"])
def test_app_server_resume_fake_protocol_sets_and_reads_exact_goal(tmp_path, goal_status):
    fake = tmp_path / "fake_app_server.py"
    fake.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    method = request['method']\n"
        "    result = None\n"
        "    if method == 'initialize':\n"
        "        sys.stdout.write(json.dumps({'method': 'server-notice'}) + '\\n' + json.dumps({'id': request['id'], 'result': {}}) + '\\n')\n"
        "        sys.stdout.flush()\n"
        "    elif method == 'initialized':\n"
        "        continue\n"
        "    elif method == 'thread/goal/set':\n"
        "        assert request['params'] == {'threadId': 'root-thread', 'status': 'active'}\n"
        "        result = {'goal': {'threadId': 'root-thread', 'status': 'active'}}\n"
        "    elif method == 'thread/goal/get':\n"
        "        assert request['params'] == {'threadId': 'root-thread'}\n"
        "        result = {'goal': {'threadId': 'root-thread', 'status': 'active'}}\n"
        "    else:\n"
        "        raise AssertionError(method)\n"
        "    print(json.dumps({'id': request['id'], 'result': result}), flush=True)\n",
        encoding="utf-8",
    )
    root = {
        "conversation_id": "root-thread",
        "codex_home": str(tmp_path / "codex-home"),
        "worktree": str(tmp_path),
        "goal_db": str(tmp_path / "goals.sqlite"),
    }
    connection = sqlite3.connect(root["goal_db"])
    connection.execute(
        "CREATE TABLE thread_goals (thread_id TEXT, goal_id TEXT, status TEXT, "
        "objective TEXT, token_budget INTEGER, updated_at_ms INTEGER)"
    )
    connection.execute(
        "INSERT INTO thread_goals VALUES (?, ?, ?, ?, ?, ?)",
        ("root-thread", "goal-1", "active", "resume objective", 11, 2),
    )
    connection.commit()
    connection.close()
    goal = {
        "goal_id": "goal-1",
        "status": goal_status,
        "objective": "resume objective",
        "token_budget": 11,
    }
    returned = poller.app_server_resume_root_goal(
        root,
        goal,
        sys.executable,
        "unused.js",
        app_server_command=[sys.executable, str(fake)],
        timeout=3.0,
    )
    assert returned == {"threadId": "root-thread", "status": "active"}


def test_pending_root_turn_does_not_block_goal_recovery_or_duplicate_wake(
    tmp_path, monkeypatch
):
    statuses = {"child": "paused"}
    _supervision_setup(monkeypatch, statuses, root_status="paused", root_goal_id="goal-pending")
    child, _ = _child(tmp_path, "child")
    queue_db = tmp_path / "root-queue.sqlite"
    connection = _queued_items_db(queue_db)
    connection.execute(
        "INSERT INTO queued_items VALUES (?, ?, ?, ?, ?, ?)",
        ("resume", "root-thread", json.dumps({"UserInput": {"content": [{"text": "/goal resume"}]}}), 0, 1, 1),
    )
    connection.commit()
    connection.close()
    registry = _registry(tmp_path, child, root_queue=queue_db)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "schema": poller.SCHEMA,
        "agents": {},
        "supervision": {"all_execution_idle": True, "wake_pending": False, "wake_sent": False},
    }), encoding="utf-8")
    resumed = []
    queued = []
    monkeypatch.setattr(poller, "queue_root", lambda *args: queued.append(args[1]))
    monkeypatch.setattr(
        poller,
        "app_server_resume_root_goal",
        lambda root, goal, node, codex, **kwargs: (
            resumed.append(goal["goal_id"]) or {"threadId": "root-thread", "status": "active"}
        ),
    )
    args = _args(registry, state)
    args.root_goal_resume_once = True
    assert poller.poll_once(args)[0] == []
    assert resumed == ["goal-pending"]
    assert queued == []
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["supervision"]["wake_pending"] is False
    assert persisted["supervision"]["root_goal_resume_consumed_goal_id"] == "goal-pending"


def test_pending_active_root_wake_persists_supervision_without_early_return(
    tmp_path, monkeypatch
):
    statuses = {"child": "paused"}
    _supervision_setup(monkeypatch, statuses, root_status="active")
    child, _ = _child(tmp_path, "child")
    queue_db = tmp_path / "root-queue.sqlite"
    connection = _queued_items_db(queue_db)
    connection.execute(
        "INSERT INTO queued_items VALUES (?, ?, ?, ?, ?, ?)",
        ("pending", "root-thread", json.dumps({"UserInput": {"content": [{"text": "turn"}]}}), 0, 1, 1),
    )
    connection.commit()
    connection.close()
    registry = _registry(tmp_path, child, root_queue=queue_db)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "schema": poller.SCHEMA,
        "agents": {},
        "supervision": {
            "all_execution_idle": True,
            "wake_pending": True,
            "wake_sent": False,
        },
    }), encoding="utf-8")
    queued = []
    monkeypatch.setattr(poller, "queue_root", lambda *args: queued.append(args[1]))
    assert poller.poll_once(_args(registry, state))[0] == []
    assert queued == []
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["supervision"]["wake_pending"] is True
    assert persisted["supervision"]["wake_sent"] is False
    assert persisted["supervision"]["root_goal_status"] == "active"
    assert persisted["supervision"]["execution_agents"]["child"]["idle"] is True


def test_decision_wake_queue_failure_retries_without_duplicate_loss(tmp_path, monkeypatch):
    statuses = {"child": "active"}
    _supervision_setup(monkeypatch, statuses)
    child, _ = _child(tmp_path, "child")
    registry = _registry(tmp_path, child)
    state = tmp_path / "state.json"
    queued = []

    def queue(root, message, node, codex):
        if message.startswith(poller.ROOT_DECISION_WAKE_PREFIX):
            raise RuntimeError("synthetic wake outage")
        queued.append(message)

    monkeypatch.setattr(poller, "queue_root", queue)
    args = _args(registry, state)
    assert poller.poll_once(args)[0] == []
    statuses["child"] = "paused"
    assert poller.poll_once(args)[0]
    assert len(queued) == 1
    with pytest.raises(RuntimeError, match="synthetic wake outage"):
        poller.poll_once(args)
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["supervision"]["wake_pending"] is True

    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    assert poller.poll_once(args)[0] == []
    assert len(queued) == 2
    assert queued[-1].startswith(poller.ROOT_DECISION_WAKE_PREFIX)


def test_goal_none_active_cpu_turn_blocks_all_idle_until_completion(tmp_path, monkeypatch):
    statuses = {"child": "NONE"}
    _supervision_setup(monkeypatch, statuses)
    child, rollout = _child(tmp_path, "child")
    rollout.write_text(_started_record("cpu"), encoding="utf-8")
    registry = _registry(tmp_path, child)
    state = tmp_path / "state.json"
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    args = _args(registry, state)

    assert poller.poll_once(args)[0] == []
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["supervision"]["all_execution_idle"] is False
    assert persisted["supervision"]["execution_agents"]["child"]["reason"] == "cpu_turn"

    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(_record("cpu"))
    assert poller.poll_once(args)[0]
    assert len(queued) == 1  # completion POLL_EVENT
    assert poller.poll_once(args)[0] == []
    assert len(queued) == 2
    assert queued[-1].startswith(poller.ROOT_DECISION_WAKE_PREFIX)


def test_legacy_cursor_is_rescanned_for_unmatched_cpu_start(tmp_path, monkeypatch):
    statuses = {"child": "NONE"}
    _supervision_setup(monkeypatch, statuses)
    child, rollout = _child(tmp_path, "child")
    rollout.write_text(_started_record("legacy"), encoding="utf-8")
    stat = rollout.stat()
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "schema": poller.SCHEMA,
                "agents": {
                    "child": {
                        "head": "same-head",
                        "goal": {"goal_id": "child-goal", "status": "NONE"},
                        "gpu_pids": [],
                        "newest_manifest": None,
                        "task_complete": [],
                        "rollout_cursor": {
                            str(rollout): {
                                "device": stat.st_dev,
                                "inode": stat.st_ino,
                                "offset": stat.st_size,
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    registry = _registry(tmp_path, child)
    assert poller.poll_once(_args(registry, state))[0] == []
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["supervision"]["all_execution_idle"] is False
    assert persisted["supervision"]["execution_agents"]["child"]["reason"] == "cpu_turn"


def test_turn_aborted_clears_own_active_cpu_turn_without_completion_id(tmp_path, monkeypatch):
    statuses = {"child": "NONE"}
    _supervision_setup(monkeypatch, statuses)
    child, rollout = _child(tmp_path, "child")
    rollout.write_text(_started_record("aborted"), encoding="utf-8")
    registry = _registry(tmp_path, child)
    state = tmp_path / "state.json"
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    args = _args(registry, state)

    assert poller.poll_once(args)[0] == []
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(_aborted_record("aborted"))
    assert poller.poll_once(args)[0] == []
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["supervision"]["all_execution_idle"] is True
    assert persisted["supervision"]["execution_agents"]["child"]["active_turns"] == []
    assert persisted["agents"]["child"]["task_complete"] == []
    assert queued and queued[-1].startswith(poller.ROOT_DECISION_WAKE_PREFIX)


def test_new_thread_start_supersedes_legacy_unterminated_starts(tmp_path, monkeypatch):
    statuses = {"child": "NONE"}
    _supervision_setup(monkeypatch, statuses)
    child, rollout = _child(tmp_path, "child")
    rollout.write_text(
        "".join(_started_record(f"legacy-{index}") for index in range(6)),
        encoding="utf-8",
    )
    registry = _registry(tmp_path, child)
    state = tmp_path / "state.json"
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    args = _args(registry, state)

    assert poller.poll_once(args)[0] == []
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["agents"]["child"]["rollout"]["active_turns"] == ["turn:legacy-5"]
    assert persisted["supervision"]["all_execution_idle"] is False

    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(_started_record("latest"))
        handle.write(_record("latest"))
    events, _ = poller.poll_once(args)
    assert events and events[0]["task_complete_added_ids"] == ["turn:latest"]
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["agents"]["child"]["rollout"]["active_turns"] == []
    assert persisted["supervision"]["all_execution_idle"] is True


def test_v4_stale_active_turns_migrate_with_one_full_rescan(tmp_path, monkeypatch):
    statuses = {"child": "NONE"}
    _supervision_setup(monkeypatch, statuses)
    child, rollout = _child(tmp_path, "child")
    old_starts = "".join(_started_record(f"stale-{index}") for index in range(6))
    rollout.write_text(old_starts + _started_record("latest") + _record("latest"), encoding="utf-8")
    stat = rollout.stat()
    stale_ids = [f"turn:stale-{index}" for index in range(6)]
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "schema": poller.SCHEMA,
                "agents": {
                    "child": {
                        "head": "same-head",
                        "goal": {"goal_id": "child-goal", "status": "NONE"},
                        "gpu_pids": [],
                        "newest_manifest": None,
                        "task_complete": ["turn:latest"],
                        "rollout_cursor": {
                            str(rollout): {
                                "device": stat.st_dev,
                                "inode": stat.st_ino,
                                "offset": stat.st_size,
                            }
                        },
                        "rollout": {
                            "status": "known",
                            "active_turns": stale_ids,
                        },
                    }
                },
                "supervision": {
                    "all_execution_idle": False,
                    "wake_pending": False,
                    "wake_sent": False,
                },
            }
        ),
        encoding="utf-8",
    )
    registry = _registry(tmp_path, child)
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )

    events, _ = poller.poll_once(_args(registry, state))
    assert events == []
    persisted = json.loads(state.read_text(encoding="utf-8"))
    rollout_state = persisted["agents"]["child"]["rollout"]
    assert rollout_state["parser_version"] == poller.ROLLOUT_PARSER_VERSION
    assert rollout_state["active_turns"] == []
    assert persisted["supervision"]["all_execution_idle"] is True
    assert queued and queued[-1].startswith(poller.ROOT_DECISION_WAKE_PREFIX)


def test_completed_turn_is_idle_but_partial_rollout_is_unknown(tmp_path, monkeypatch):
    statuses = {"child": "NONE"}
    _supervision_setup(monkeypatch, statuses)
    child, rollout = _child(tmp_path, "child")
    rollout.write_text(_started_record("done") + _record("done"), encoding="utf-8")
    registry = _registry(tmp_path, child)
    state = tmp_path / "state.json"
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    args = _args(registry, state)

    assert poller.poll_once(args)[0] == []
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["supervision"]["all_execution_idle"] is True
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(_started_record("partial").rstrip("\n"))
    assert poller.poll_once(args)[0] == []
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["supervision"]["all_execution_idle"] is False
    assert persisted["supervision"]["execution_agents"]["child"]["reason"] == "rollout_unknown"
    assert queued == []


def test_unavailable_rollout_is_conservatively_non_idle(tmp_path, monkeypatch):
    statuses = {"child": "NONE"}
    _supervision_setup(monkeypatch, statuses)
    child, rollout = _child(tmp_path, "child")
    rollout.write_text(_started_record("done") + _record("done"), encoding="utf-8")
    registry = _registry(tmp_path, child)
    state = tmp_path / "state.json"
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    args = _args(registry, state)

    assert poller.poll_once(args)[0] == []
    rollout.unlink()
    assert poller.poll_once(args)[0] == []
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["supervision"]["all_execution_idle"] is False
    assert persisted["supervision"]["execution_agents"]["child"]["reason"] == "rollout_unknown"
    assert queued == []


def test_transient_gpu_pid_survives_queue_failure_and_deferred_merge(tmp_path, monkeypatch):
    _setup(monkeypatch)
    child, _ = _child(tmp_path, "child")
    registry = _registry(tmp_path, child)
    state = tmp_path / "state.json"
    args = _args(registry, state)
    head = {"value": "same-head"}
    gpu = {"value": []}
    monkeypatch.setattr(poller, "git_head", lambda worktree: head["value"])
    monkeypatch.setattr(poller, "gpu_pids", lambda: list(gpu["value"]))
    monkeypatch.setattr(poller, "owned_gpu_pids", lambda worktree, pids: list(pids))
    assert poller.poll_once(args)[0] == []

    gpu["value"] = [123]

    def fail_queue(root, message, node, codex):
        raise RuntimeError("synthetic queue outage")

    monkeypatch.setattr(poller, "queue_root", fail_queue)
    with pytest.raises(RuntimeError, match="synthetic queue outage"):
        poller.poll_once(args)
    failed = json.loads(state.read_text(encoding="utf-8"))
    event = failed["deferred_events"][0]
    assert event["observed_gpu_pids"] == [123]
    assert failed["pending_snapshots"]["child"]["gpu_pids"] == [123]
    assert failed["agents"]["child"]["gpu_pids"] == []

    # The GPU disappears before retry and another field changes.  The merged
    # event's latest snapshot may be GPU-free, but the observed PID remains
    # auditable until the deferred event is delivered.
    gpu["value"] = []
    head["value"] = "new-head"
    queued = []
    monkeypatch.setattr(
        poller, "queue_root", lambda root, message, node, codex: queued.append(message)
    )
    events, _ = poller.poll_once(args)
    assert len(events) == 1
    payload = json.loads(queued[0].split("\n", 1)[1].split("\nReview", 1)[0])
    assert payload["observed_gpu_pids"] == [123]
    delivered = json.loads(state.read_text(encoding="utf-8"))
    assert delivered["agents"]["child"]["head"] == "new-head"
    assert delivered["agents"]["child"]["gpu_pids"] == []
    assert "deferred_events" not in delivered
