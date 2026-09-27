from __future__ import annotations

import json
from pathlib import Path

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
                "agent_eval": {"runtime_agent_key": "agent_c1_validation"},
                "agent_cm": {"runtime_agent_key": "agent_cm_temporal"},
            }
        }),
        encoding="utf-8",
    )
    registry = {
        "runtime_bindings": {"path": ".runtime/AGENT_BINDINGS.json"},
        "agents": [
            {"agent_key": "root"},
            {"agent_key": "agent_baseline"},
            {"agent_key": "agent_c1_validation"},
            {"agent_key": "agent_cm_temporal"},
            {"agent_key": "agent_workflow"},
        ],
    }
    selected = events.worker_agents(registry, tmp_path)
    assert [item["agent_key"] for item in selected] == [
        "agent_c1_validation", "agent_cm_temporal", "agent_workflow"
    ]


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
