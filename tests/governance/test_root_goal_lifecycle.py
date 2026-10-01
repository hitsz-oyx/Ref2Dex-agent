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


def test_recovery_cycle_contract_is_documented_for_watchdog_and_poller() -> None:
    coordination = (ROOT / "docs/AGENT_COORDINATION.md").read_text(encoding="utf-8")
    poller = (ROOT / "docs/AGENT_POLLER.md").read_text(encoding="utf-8")
    for document in (coordination, poller):
        compact = re.sub(r"\s+", "", document)
        assert "连续的`paused`或`blocked`周期" in compact or "一个连续的`paused`或`blocked`状态周期" in compact
        assert (
            "被重新观察为`active`" in compact
            or "回到`active`才清除" in compact
        )
        assert "清除" in compact
        assert "再次进入`paused`或`blocked`" in compact or "之后再次`paused`或`blocked`" in compact
        assert "不表示研究任务完成" in compact or "不把`paused`、`blocked`或恢复失败解释为科研任务完成" in compact


def test_root_recovery_ownership_and_legacy_poller_opt_in_are_documented() -> None:
    coordination = (ROOT / "docs/AGENT_COORDINATION.md").read_text(encoding="utf-8")
    poller_doc = (ROOT / "docs/AGENT_POLLER.md").read_text(encoding="utf-8")
    root_policy = (ROOT / "docs/ROOT_AGENT.md").read_text(encoding="utf-8")
    poller_script = (ROOT / "scripts/runtime_support.py").read_text(encoding="utf-8")
    compatibility_entry = (ROOT / "scripts/agent_result_poller.py").read_text(encoding="utf-8")

    assert "poller 不负责 root liveness" in poller_doc
    assert "worker_event_poller.py` 是当前 Broker 事件源" in poller_doc
    assert "agent_result_poller.py` 只作为历史兼容入口保留" in poller_doc
    assert "--root-goal-resume-once" in poller_doc
    assert "默认的 lease-authorized" in poller_doc
    assert "root_watchdog.py" in root_policy
    assert "allow_root_resume" in coordination
    assert "--root-goal-resume-once" in poller_script
    assert "runtime_support" in compatibility_entry


def test_autonomous_decision_memo_replaces_mandatory_option_pair() -> None:
    policy = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    root_policy = (ROOT / "docs/ROOT_AGENT.md").read_text(encoding="utf-8")

    assert "root 自主选择并执行" in policy
    assert "预设的 Option A/Option B 之间插入选择" in policy
    assert "root 先记录简短 Decision Memo" in root_policy
    assert "历史处置记录不构成永久的 Option A/Option B 插入点" in root_policy
    assert "* Option A；" not in policy
    assert "* Option B" not in policy


def test_recovery_control_plane_has_single_owner_and_shared_decision_helper() -> None:
    coordination = (ROOT / "docs/AGENT_COORDINATION.md").read_text(encoding="utf-8")
    broker = (ROOT / "docs/AGENT_BROKER.md").read_text(encoding="utf-8")
    poller = (ROOT / "docs/AGENT_POLLER.md").read_text(encoding="utf-8")
    watchdog = (ROOT / "scripts/root_watchdog.py").read_text(encoding="utf-8")
    legacy = (ROOT / "scripts/runtime_support.py").read_text(encoding="utf-8")

    for document in (coordination, broker, poller):
        assert "root_watchdog.py" in document
        assert "recovery_control_decision" in document or "默认恢复 owner" in document
        assert "usage_limited" in document
        assert "budget_limited" in document
    assert "recovery_control_decision" in watchdog
    assert "recovery_control_decision" in legacy
    assert "ROOT_TERMINAL_GOAL_STATUSES" in legacy
