from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from scripts import worker_event_poller as events
from scripts.researchctl import main as researchctl_main
from scripts.root_watchdog import classify


ROOT = Path(__file__).resolve().parents[1]


def test_registry_declares_fixed_pool_and_local_bindings() -> None:
    registry = json.loads((ROOT / "docs/AGENT_REGISTRY.json").read_text(encoding="utf-8"))
    roles = {item["agent_key"]: item for item in registry["pool"]["roles"]}
    assert set(roles) == {"root", "agent_cm", "agent_rl", "agent_eval", "agent_infra"}
    assert registry["pool"]["dynamic_creation"] == "user_approval_only"
    assert registry["pool"]["nested_delegation"] == "forbidden"
    assert registry["runtime_bindings"]["path"] == ".runtime/AGENT_BINDINGS.json"


def test_worker_selection_prefers_one_runtime_binding_per_role(tmp_path: Path) -> None:
    (tmp_path / ".runtime").mkdir()
    (tmp_path / ".runtime/AGENT_BINDINGS.json").write_text(
        json.dumps({
            "bindings": {
                "agent_eval": {
                    "runtime_agent_key": "agent_c1_validation",
                    "status": "bound",
                    "conversation_id": "eval-thread",
                },
                "agent_cm": {
                    "runtime_agent_key": "agent_cm_temporal",
                    "status": "bound",
                    "conversation_id": "cm-thread",
                },
            }
        }),
        encoding="utf-8",
    )
    registry = {
        "runtime_bindings": {"path": ".runtime/AGENT_BINDINGS.json"},
        "agents": [
            {"agent_key": "root"},
            {"agent_key": "agent_baseline"},
            {"agent_key": "agent_c1_validation", "conversation_id": "eval-thread"},
            {"agent_key": "agent_cm_temporal", "conversation_id": "cm-thread"},
            {"agent_key": "agent_workflow", "role_key": "agent_infra"},
        ],
    }
    selected = events.worker_agents(registry, tmp_path)
    assert {item["agent_key"] for item in selected} == {
        "agent_c1_validation", "agent_cm_temporal"
    }


def test_worker_selection_fails_closed_on_identity_mismatch_and_retiring_record(
    tmp_path: Path,
) -> None:
    (tmp_path / ".runtime").mkdir()
    (tmp_path / ".runtime/AGENT_BINDINGS.json").write_text(
        json.dumps({
            "bindings": {
                "agent_eval": {
                    "runtime_agent_key": "legacy",
                    "status": "bound",
                    "conversation_id": "new-thread",
                },
                "agent_infra": {
                    "runtime_agent_key": "retiring",
                    "status": "bound",
                },
            }
        }),
        encoding="utf-8",
    )
    registry = {
        "runtime_bindings": {"path": ".runtime/AGENT_BINDINGS.json"},
        "agents": [
            {"agent_key": "root"},
            # The identity mismatch must not silently select this historical
            # runtime or another record for the same role.
            {"agent_key": "legacy", "role_key": "agent_eval", "conversation_id": "old-thread"},
            {"agent_key": "retiring", "role_key": "agent_infra", "lifecycle": "retiring"},
            {"agent_key": "unbound", "role_key": "agent_eval", "conversation_id": "new-thread"},
        ],
    }
    assert events.worker_agents(registry, tmp_path) == []


def test_worker_event_deduplicates_shared_canonical_manifest(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".runtime").mkdir()
    child_a = tmp_path / "child-a"
    child_b = tmp_path / "child-b"
    child_a.mkdir()
    child_b.mkdir()
    shared = tmp_path / "shared" / "run_manifest.json"
    shared.parent.mkdir()
    shared.write_text('{"version": 1}\n', encoding="utf-8")
    target_a = tmp_path / "target-a"
    target_a.mkdir()
    (target_a / "run_manifest.json").symlink_to(shared)
    (child_a / "outputs").symlink_to(target_a, target_is_directory=True)
    target_b = tmp_path / "target-b" / "nested"
    target_b.mkdir(parents=True)
    (target_b / "run_manifest.json").symlink_to(shared)
    (child_b / "outputs").symlink_to(target_b.parent, target_is_directory=True)

    agents = [
        {
            "agent_key": "root",
            "conversation_id": "root-thread",
        },
        {
            "agent_key": "worker-a",
            "role_key": "agent_eval",
            "conversation_id": "a-thread",
            "codex_home": str(tmp_path / "a-home"),
            "worktree": str(child_a),
            "branch": "agent/eval",
        },
        {
            "agent_key": "worker-b",
            "role_key": "agent_infra",
            "conversation_id": "b-thread",
            "codex_home": str(tmp_path / "b-home"),
            "worktree": str(child_b),
            "branch": "agent/infra",
        },
    ]
    (tmp_path / ".runtime/AGENT_BINDINGS.json").write_text(
        json.dumps({
            "bindings": {
                "agent_eval": {"runtime_agent_key": "worker-a", "status": "bound"},
                "agent_infra": {"runtime_agent_key": "worker-b", "status": "bound"},
            }
        }),
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    registry = tmp_path / "docs" / "registry.json"
    registry.write_text(json.dumps({
        "runtime_bindings": {"path": ".runtime/AGENT_BINDINGS.json"},
        "agents": agents,
    }), encoding="utf-8")
    state = tmp_path / "state.json"
    queued = []
    monkeypatch.setattr(events.legacy, "gpu_pids", lambda: [])
    monkeypatch.setattr(events.legacy, "owned_gpu_pids", lambda worktree, pids: [])
    monkeypatch.setattr(events.legacy, "git_head", lambda worktree: "same-head")
    monkeypatch.setattr(events.legacy, "goal_state", lambda agent: {"goal_id": None, "status": "NONE"})

    def fake_snapshot(agent, all_gpu_pids, previous=None):
        return {
            "head": "same-head",
            "goal": {"goal_id": None, "status": "NONE"},
            "gpu_pids": [],
            "newest_manifest": events.legacy.newest_manifest(agent["worktree"]),
            "task_complete": [],
            "rollout_cursor": {},
            "rollout": {"status": "known", "active_turns": []},
        }

    monkeypatch.setattr(events.legacy, "snapshot", fake_snapshot)
    monkeypatch.setattr(events.legacy, "queue_root", lambda root, message, node, codex: queued.append(message))
    args = SimpleNamespace(
        registry=str(registry), state=str(state), dry_run=False,
        codex_node="node", codex_js="codex.js",
    )
    assert events.poll_once(args)[0] == []
    old_mtime = shared.stat().st_mtime_ns
    shared.write_text('{"version": 2}\n', encoding="utf-8")
    os.utime(shared, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))
    detected, _ = events.poll_once(args)
    assert len(detected) == 1
    assert detected[0]["changed"] == ["newest_manifest"]
    assert len(queued) == 1


def test_watchdog_classification_never_treats_unknown_rollout_as_idle() -> None:
    status, reason, idle_since = classify(
        {"status": "active"},
        {"rollout": {"status": "unknown", "active_turns": []}},
        False,
        1000.0,
        None,
        1.0,
    )
    assert status == "ROOT_RUNNING"
    assert "active turn" in reason
    assert idle_since is None


def test_watchdog_budget_and_terminal_states_are_not_resumed() -> None:
    assert classify(
        {"status": "budget_limited"},
        {"rollout": {"status": "known", "active_turns": []}},
        False,
        1000.0,
        None,
        1.0,
    )[0] == "ROOT_BUDGET_LIMITED"
    assert classify(
        {"status": "paused"},
        {"rollout": {"status": "known", "active_turns": []}},
        False,
        1000.0,
        None,
        1.0,
    )[0] == "ROOT_PAUSED"


def test_researchctl_pause_and_resume_only_change_local_lease(tmp_path: Path) -> None:
    lease = tmp_path / "SUPERVISOR_LEASE.json"
    assert researchctl_main(["supervisor", "resume", "--lease", str(lease)]) == 0
    resumed = json.loads(lease.read_text(encoding="utf-8"))
    assert resumed["enabled"] is True
    assert resumed["allow_root_resume"] is True
    assert researchctl_main(["supervisor", "pause", "--lease", str(lease)]) == 0
    paused = json.loads(lease.read_text(encoding="utf-8"))
    assert paused["enabled"] is False
    assert paused["generation"] == resumed["generation"] + 1
