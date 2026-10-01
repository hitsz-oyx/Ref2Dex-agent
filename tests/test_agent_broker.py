from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.agent_broker import AgentBroker, BrokerError


ROOT = Path(__file__).resolve().parents[1]


def broker(tmp_path: Path, *, mutate=None) -> AgentBroker:
    bindings = json.loads((ROOT / ".runtime/AGENT_BINDINGS.json").read_text(encoding="utf-8"))
    # Keep the fixture independent of the machine-local runtime binding.  The
    # real agent_cm binding may be present while this test needs an explicitly
    # malformed binding to exercise the fail-closed path.
    bindings["bindings"]["agent_cm"].update({"status": "bound"})
    bindings["bindings"]["agent_cm"].pop("runtime_agent_key", None)
    bindings["bindings"]["agent_infra"].update(
        {"runtime_agent_key": "infra-test", "provider": "other", "profile": "test"}
    )
    if mutate is not None:
        mutate(bindings["bindings"]["agent_infra"])
    binding_path = tmp_path / "bindings.json"
    binding_path.write_text(json.dumps(bindings), encoding="utf-8")
    return AgentBroker(
        tmp_path / "tasks.sqlite",
        tmp_path / "state.sqlite",
        ROOT / "docs/AGENT_ROLES.yaml",
        binding_path,
    )


def test_broker_persists_four_message_contract_and_handoff(tmp_path: Path) -> None:
    instance = broker(tmp_path)
    instance.set_desired_state("RUNNING", reason="test")
    dispatched = instance.dispatch(
        task_id="T-test-broker",
        target_agent="agent_infra",
        objective="verify broker state transitions",
        constraints={"gpu": 0},
        done_when=["handoff"],
    )
    claimed = instance.claim("agent_infra", lease_seconds=30)
    assert claimed and claimed["task_id"] == dispatched["task_id"]
    instance.update(
        task_id=claimed["task_id"], agent_key="agent_infra", lease_token=claimed["lease_token"],
        progress={"tests": 1},
    )
    instance.handoff(
        task_id=claimed["task_id"], agent_key="agent_infra", status="COMPLETED",
        result={"label": "PROMISING"}, evidence=["smoke"], lease_token=claimed["lease_token"],
    )
    instance.control(action="RESUME", target="root", reason="test")
    status = instance.status()
    assert [item["type"] for item in instance.messages_since()] == [
        "TASK_DISPATCH", "TASK_UPDATE", "TASK_HANDOFF", "CONTROL"
    ]
    assert status["tasks"][0]["status"] == "COMPLETED"
    assert status["supervisor"]["desired_state"] == "RUNNING"


def test_broker_rejects_dynamic_or_unbound_targets(tmp_path: Path) -> None:
    instance = broker(tmp_path)
    with pytest.raises(BrokerError, match="unknown fixed agent_key"):
        instance.dispatch(task_id="T-dynamic", target_agent="agent_17", objective="no")
    with pytest.raises(BrokerError, match="no runtime binding"):
        instance.dispatch(task_id="T-unbound", target_agent="agent_cm", objective="no")


def test_broker_rejects_wrong_lease_owner(tmp_path: Path) -> None:
    instance = broker(tmp_path)
    instance.set_desired_state("RUNNING", reason="test")
    instance.dispatch(task_id="T-lease", target_agent="agent_infra", objective="lease")
    claimed = instance.claim("agent_infra")
    assert claimed
    with pytest.raises(BrokerError, match="invalid task lease token"):
        instance.update(task_id="T-lease", agent_key="agent_infra", lease_token="wrong")


def test_pause_blocks_new_dispatch_and_claim_but_allows_in_flight_handoff(tmp_path: Path) -> None:
    instance = broker(tmp_path)
    instance.set_desired_state("RUNNING", reason="test")
    instance.dispatch(task_id="T-in-flight", target_agent="agent_infra", objective="finish")
    claimed = instance.claim("agent_infra")
    assert claimed
    instance.control(action="PAUSE", target="root", reason="user pause")

    with pytest.raises(BrokerError, match="supervisor is PAUSED"):
        instance.dispatch(task_id="T-blocked", target_agent="agent_infra", objective="blocked")
    with pytest.raises(BrokerError, match="supervisor is PAUSED"):
        instance.claim("agent_infra")
    instance.handoff(
        task_id="T-in-flight", agent_key="agent_infra", status="COMPLETED",
        result={"ok": True}, lease_token=claimed["lease_token"],
    )


def test_update_and_handoff_require_live_lease(tmp_path: Path) -> None:
    instance = broker(tmp_path)
    instance.set_desired_state("RUNNING", reason="test")
    instance.dispatch(task_id="T-token", target_agent="agent_infra", objective="token")
    claimed = instance.claim("agent_infra", lease_seconds=30)
    assert claimed

    with pytest.raises(BrokerError, match="lease token is required"):
        instance.update(task_id="T-token", agent_key="agent_infra")
    with pytest.raises(BrokerError, match="lease token is required"):
        instance.handoff(task_id="T-token", agent_key="agent_infra", status="COMPLETED")


def test_expired_lease_and_terminal_transition_are_rejected(tmp_path: Path) -> None:
    instance = broker(tmp_path)
    instance.set_desired_state("RUNNING", reason="test")
    instance.dispatch(task_id="T-expired", target_agent="agent_infra", objective="expiry")
    claimed = instance.claim("agent_infra", lease_seconds=1)
    assert claimed
    with instance._connect(instance.tasks_db) as db:
        db.execute("UPDATE tasks SET lease_expires_at=0 WHERE task_id='T-expired'")
        db.commit()
    with pytest.raises(BrokerError, match="lease has expired"):
        instance.update(
            task_id="T-expired", agent_key="agent_infra", lease_token=claimed["lease_token"]
        )


def test_dispatch_requires_running_supervisor(tmp_path: Path) -> None:
    instance = broker(tmp_path)
    assert instance.status()["supervisor"]["desired_state"] == "PAUSED"
    with pytest.raises(BrokerError, match="supervisor is PAUSED"):
        instance.dispatch(task_id="T-paused", target_agent="agent_infra", objective="blocked")


@pytest.mark.parametrize(
    "mutate,pattern",
    [
        (lambda binding: binding.update(status="retiring"), "not bound"),
        (lambda binding: binding.update(status="unbound_after_cleanup"), "not bound"),
        (lambda binding: binding.pop("runtime_agent_key"), "no runtime binding"),
        (lambda binding: binding.update(provider="unsupported"), "unsupported provider"),
        (lambda binding: binding.update(branch="main"), "branch mismatch"),
    ],
)
def test_dispatch_rejects_invalid_runtime_binding_before_persisting(
    tmp_path: Path, mutate, pattern: str,
) -> None:
    instance = broker(tmp_path, mutate=mutate)
    with pytest.raises(BrokerError, match=pattern):
        instance.dispatch(
            task_id="T-invalid-binding",
            target_agent="agent_infra",
            objective="must not be persisted",
        )
    status = instance.status()
    assert status["tasks"] == []
    assert status["messages"] == []
