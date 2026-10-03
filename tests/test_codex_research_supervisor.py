from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.codex_research_supervisor import (
    CapacityWatchdog,
    SessionState,
    build_parser,
    consume_rollout_events,
)


def _home(tmp_path: Path, name: str, *, updated_ms: int, archived: int = 0, events=()) -> Path:
    home = tmp_path / name
    home.mkdir()
    rollout = home / "rollout.jsonl"
    rollout.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    state = sqlite3.connect(home / "state_5.sqlite")
    state.execute(
        "CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT, "
        "updated_at_ms INTEGER, archived INTEGER)"
    )
    state.execute(
        "INSERT INTO threads VALUES (?, ?, ?, ?)",
        (name + "-thread", str(rollout), updated_ms, archived),
    )
    state.commit()
    state.close()

    queue = sqlite3.connect(home / "queue_1.sqlite")
    queue.execute(
        "CREATE TABLE queued_items (id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, "
        "payload_json TEXT NOT NULL)"
    )
    queue.commit()
    queue.close()
    return home


def _event(kind: str, *, text: str | None = None, turn_id: str = "turn", error=None) -> dict:
    payload = {"type": kind, "turn_id": turn_id}
    if text:
        payload["error"] = {"message": text, "codex_error_info": "server_overloaded"}
    if error is not None:
        payload["error"] = error
    return {"payload": payload}


def _watchdog(tmp_path: Path):
    return CapacityWatchdog(
        build_parser(
            [
                "--scan-root",
                str(tmp_path),
                "--state",
                str(tmp_path / "watchdog.json"),
                "--active-window",
                "86400",
                "--interval",
                "60",
                "--dry-run",
            ]
        )
    )


def test_rollout_reader_keeps_partial_records_for_next_cycle(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(json.dumps(_event("task_started")) + "\n{", encoding="utf-8")
    state = SessionState()
    assert len(consume_rollout_events(state, str(rollout))) == 1
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write('"payload": {"type": "task_complete"}}\n')
    assert len(consume_rollout_events(state, str(rollout))) == 1


def test_capacity_event_arms_and_sends_continue_once_per_minute(tmp_path):
    now = 2_000_000_000.0
    _home(
        tmp_path,
        ".codex_one",
        updated_ms=int(now * 1000),
        events=[_event("task_complete", text="Selected model is at capacity")],
    )
    watchdog = _watchdog(tmp_path)
    assert watchdog.cycle(now=now)["sent"] == []
    assert watchdog.cycle(now=now + 59)["sent"] == []
    first = watchdog.cycle(now=now + 60)
    assert first["scanned"] == 1 and len(first["sent"]) == 1
    assert watchdog.cycle(now=now + 120)["sent"] == []


def test_capacity_phrase_in_normal_rollout_text_does_not_arm(tmp_path):
    now = 2_000_000_000.0
    _home(
        tmp_path,
        ".codex_one",
        updated_ms=int(now * 1000),
        events=[
            {"payload": {"type": "assistant_message", "text": "Selected model is at capacity"}},
            {"payload": {"type": "custom_tool_call", "command": "server_overloaded"}},
        ],
    )
    result = _watchdog(tmp_path).cycle(now=now + 60)
    assert result["armed"] == []
    assert result["sent"] == []


def test_new_turn_clears_capacity_arm(tmp_path):
    now = 2_000_000_000.0
    home = _home(
        tmp_path,
        ".codex_one",
        updated_ms=int(now * 1000),
        events=[_event("task_complete", text="server_overloaded")],
    )
    watchdog = _watchdog(tmp_path)
    assert watchdog.cycle(now=now)["sent"] == []
    assert watchdog.cycle(now=now + 60)["sent"]
    rollout = home / "rollout.jsonl"
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_event("task_started", turn_id="recovery")) + "\n")
    assert watchdog.cycle(now=now + 61)["armed"] == []


def test_stale_archived_and_old_capacity_sessions_are_ignored(tmp_path):
    now = 2_000_000_000.0
    _home(
        tmp_path,
        ".codex_stale",
        updated_ms=int((now - 90_000) * 1000),
        events=[_event("task_complete", text="Selected model is at capacity")],
    )
    _home(
        tmp_path,
        ".codex_archived",
        updated_ms=int(now * 1000),
        archived=1,
        events=[_event("task_complete", text="Selected model is at capacity")],
    )
    _home(
        tmp_path,
        ".codex_old_error",
        updated_ms=int(now * 1000),
        events=[
            {
                "timestamp": "2020-01-01T00:00:00Z",
                **_event("task_complete", text="Selected model is at capacity"),
            }
        ],
    )
    result = _watchdog(tmp_path).cycle(now=now)
    assert result["scanned"] == 1
    assert result["sent"] == []


def test_nonterminal_capacity_text_does_not_arm(tmp_path):
    now = 2_000_000_000.0
    _home(
        tmp_path,
        ".codex_one",
        updated_ms=int(now * 1000),
        events=[
            _event("task_started", turn_id="active"),
            _event("response_error", text="Selected model is at capacity", turn_id="active"),
        ],
    )
    result = _watchdog(tmp_path).cycle(now=now)
    assert result["armed"] == [] and result["sent"] == []


def test_subagent_threads_are_scanned_but_never_queued(tmp_path):
    now = 2_000_000_000.0
    home = _home(
        tmp_path,
        ".codex_subagent",
        updated_ms=int(now * 1000),
        events=[_event("task_complete", text="Selected model is at capacity")],
    )
    state = sqlite3.connect(home / "state_5.sqlite")
    state.execute("ALTER TABLE threads ADD COLUMN thread_source TEXT")
    state.execute("ALTER TABLE threads ADD COLUMN agent_path TEXT")
    state.execute("UPDATE threads SET thread_source='subagent', agent_path='/root/child'")
    state.commit()
    state.close()
    result = _watchdog(tmp_path).cycle(now=now)
    assert result["scanned"] == 1
    assert result["eligible"] == 0
    assert result["skipped_subagents"] == 1
    assert result["sent"] == []
