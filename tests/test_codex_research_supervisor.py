import json
import sqlite3

from scripts.codex_research_supervisor import (
    SupervisorState,
    apply_activity_event,
    build_parser,
    consume_rollout_events,
    ResearchSupervisor,
)


THREAD = "01a0d943-de74-7021-8d50-2a4e87fde613"


def _codex_home(tmp_path, rollout, goal_status="active"):
    home = tmp_path / "codex"
    home.mkdir()
    state = sqlite3.connect(home / "state_5.sqlite")
    state.execute(
        "CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT, archived INTEGER, "
        "title TEXT, cwd TEXT)"
    )
    state.execute(
        "INSERT INTO threads VALUES (?, ?, ?, ?, ?)",
        (THREAD, str(rollout), 0, "root", str(tmp_path)),
    )
    state.commit()
    state.close()

    queue = sqlite3.connect(home / "queue_1.sqlite")
    queue.execute(
        "CREATE TABLE queued_items (id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, "
        "payload_json TEXT NOT NULL, queue_order INTEGER NOT NULL, "
        "created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL)"
    )
    queue.commit()
    queue.close()

    goals = sqlite3.connect(home / "goals_1.sqlite")
    goals.execute(
        "CREATE TABLE thread_goals (thread_id TEXT PRIMARY KEY, status TEXT NOT NULL)"
    )
    goals.execute("INSERT INTO thread_goals VALUES (?, ?)", (THREAD, goal_status))
    goals.commit()
    goals.close()
    return home


def _event(event_type, turn_id, root_turn_id=None):
    payload = {"type": event_type, "turn_id": turn_id}
    if root_turn_id is not None:
        payload["root_turn_id"] = root_turn_id
    return {"payload": payload}


def test_activity_ignores_child_turns():
    active, effect = apply_activity_event(None, _event("task_started", "root"))
    assert (active, effect) == ("root", "root_started")

    active, effect = apply_activity_event(
        active, _event("task_started", "child", root_turn_id="root")
    )
    assert (active, effect) == ("root", None)
    active, effect = apply_activity_event(
        active, _event("task_complete", "child", root_turn_id="root")
    )
    assert (active, effect) == ("root", None)

    active, effect = apply_activity_event(active, _event("task_complete", "root"))
    assert (active, effect) == (None, "root_finished")


def test_consume_rollout_replays_complete_records_and_retries_partial(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        json.dumps(_event("task_started", "root")) + "\n" + '{"payload":',
        encoding="utf-8",
    )
    state = SupervisorState(THREAD)
    events = consume_rollout_events(state, rollout)
    assert len(events) == 1
    state.active_turn_id, _ = apply_activity_event(state.active_turn_id, events[0])
    assert state.active_turn_id == "root"

    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "task_complete", "turn_id": "root"}) + "}\n")
    events = consume_rollout_events(state, rollout)
    assert len(events) == 1
    state.active_turn_id, effect = apply_activity_event(state.active_turn_id, events[0])
    assert (state.active_turn_id, effect) == (None, "root_finished")


def test_dry_run_wakes_only_idle_thread(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(json.dumps(_event("task_complete", "old")) + "\n", encoding="utf-8")
    home = _codex_home(tmp_path, rollout)
    state_file = tmp_path / "supervisor.json"
    args = build_parser(
        [
            "--thread",
            THREAD,
            "--codex-home",
            str(home),
            "--codex-bin",
            "/bin/true",
            "--state-file",
            str(state_file),
            "--lock-file",
            str(tmp_path / "lock"),
            "--start-immediately",
            "--dry-run",
        ]
    )
    supervisor = ResearchSupervisor(args)
    assert supervisor.cycle(now=100.0) is False
    assert supervisor.state.wake_count == 0


def test_active_thread_is_not_woken(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(json.dumps(_event("task_started", "current")) + "\n", encoding="utf-8")
    home = _codex_home(tmp_path, rollout)
    args = build_parser(
        [
            "--thread",
            THREAD,
            "--codex-home",
            str(home),
            "--codex-bin",
            "/bin/true",
            "--state-file",
            str(tmp_path / "supervisor.json"),
            "--lock-file",
            str(tmp_path / "lock"),
            "--start-immediately",
        ]
    )
    supervisor = ResearchSupervisor(args)
    assert supervisor.cycle(now=100.0) is False
    assert supervisor.state.wake_count == 0
    assert supervisor.state.active_turn_id == "current"


def test_real_queue_is_rate_limited_and_records_wake(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(json.dumps(_event("task_complete", "old")) + "\n", encoding="utf-8")
    home = _codex_home(tmp_path, rollout)
    args = build_parser(
        [
            "--thread",
            THREAD,
            "--codex-home",
            str(home),
            "--codex-bin",
            "/bin/true",
            "--state-file",
            str(tmp_path / "supervisor.json"),
            "--lock-file",
            str(tmp_path / "lock"),
            "--interval",
            "300",
            "--start-immediately",
            "--legacy-fixed-message",
            "--message",
            "legacy-test-message",
        ]
    )
    supervisor = ResearchSupervisor(args)
    assert supervisor.cycle(now=100.0) is True
    assert supervisor.state.wake_count == 1
    assert supervisor.state.waiting_for_turn is True
    assert supervisor.cycle(now=101.0) is False


def test_blocked_goal_requires_explicit_opt_in(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(json.dumps(_event("task_complete", "old")) + "\n", encoding="utf-8")
    home = _codex_home(tmp_path, rollout, goal_status="blocked")
    common = [
        "--thread",
        THREAD,
        "--codex-home",
        str(home),
        "--codex-bin",
        "/bin/true",
        "--state-file",
        str(tmp_path / "supervisor.json"),
        "--lock-file",
        str(tmp_path / "lock"),
        "--start-immediately",
    ]
    supervisor = ResearchSupervisor(build_parser(common))
    assert supervisor.cycle(now=100.0) is False
    assert supervisor.state.wake_count == 0

    supervisor = ResearchSupervisor(
        build_parser(
            common
            + [
                "--allow-blocked",
                "--legacy-fixed-message",
                "--message",
                "legacy-test-message",
            ]
        )
    )
    assert supervisor.cycle(now=100.0) is True


def test_real_queue_is_retired_without_explicit_legacy_opt_in(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(json.dumps(_event("task_complete", "old")) + "\n", encoding="utf-8")
    home = _codex_home(tmp_path, rollout)
    args = build_parser(
        [
            "--thread",
            THREAD,
            "--codex-home",
            str(home),
            "--codex-bin",
            "/does/not/exist",
            "--state-file",
            str(tmp_path / "supervisor.json"),
            "--lock-file",
            str(tmp_path / "lock"),
            "--start-immediately",
        ]
    )
    supervisor = ResearchSupervisor(args)
    assert supervisor.run() == 0
    assert supervisor.state.wake_count == 0
