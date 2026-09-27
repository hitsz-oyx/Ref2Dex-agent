"""Regression tests for the persistent root-supervision Goal contract."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_persistent_root_goal_contract_is_machine_readable_and_documented() -> None:
    registry = json.loads((ROOT / "docs/AGENT_REGISTRY.json").read_text(encoding="utf-8"))
    lifecycle = registry["supervision"]["root_goal_lifecycle"]

    assert lifecycle["anchor"] == "persistent_supervision"
    assert lifecycle["default_status"] == "active"
    assert lifecycle["status_change_authority"] == [
        "explicit_user_pause_stop_or_switch",
        "explicit_platform_usage_budget_or_lifecycle_limit",
    ]
    assert lifecycle["child_goal_terminal_does_not_end_root"] is True
    assert lifecycle["idle_or_milestone_does_not_end_root"] is True
    assert lifecycle["blocked_status_rule"] == (
        "platform_three_consecutive_repeated_blockers_only"
    )
    assert lifecycle["after_child_completion"] == "return_to_root_task_selection_loop"

    root_policy = (ROOT / "docs/ROOT_AGENT.md").read_text(encoding="utf-8")
    coordination = (ROOT / "docs/AGENT_COORDINATION.md").read_text(encoding="utf-8")
    coordination_compact = re.sub(r"\s+", "", coordination)

    assert "root Goal；该 Goal 是恒定的生命周期锚点" in root_policy
    assert "child 终态不会结束 root Goal" in root_policy
    assert "root 都回到审计和任务选择循环" in root_policy
    assert "root Goal 是持续监督的恒定锚点" in coordination
    assert "child完成后root必须回到任务选择循环" in coordination_compact
