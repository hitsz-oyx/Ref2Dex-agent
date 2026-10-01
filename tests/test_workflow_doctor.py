from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.agent_broker import AgentBroker
from scripts.workflow_doctor import inspect_workflow


ROOT = Path(__file__).resolve().parents[1]


def _fixtures(tmp_path: Path) -> tuple[AgentBroker, Path, Path]:
    bindings = json.loads((ROOT / ".runtime/AGENT_BINDINGS.json").read_text(encoding="utf-8"))
    bindings["bindings"]["agent_infra"].update(
        {"status": "bound", "runtime_agent_key": "infra-test", "provider": "other", "profile": "test"}
    )
    binding_path = tmp_path / "bindings.json"
    binding_path.write_text(json.dumps(bindings), encoding="utf-8")
    broker = AgentBroker(
        tmp_path / "tasks.sqlite", tmp_path / "state.sqlite",
        ROOT / "docs/AGENT_ROLES.yaml", binding_path,
    )
    return broker, binding_path, tmp_path / "lease.json"


def test_doctor_reports_optional_unbound_role_and_valid_schema(tmp_path: Path) -> None:
    broker, binding_path, lease_path = _fixtures(tmp_path)
    lease_path.write_text(json.dumps({"root_agent": "root", "enabled": False}), encoding="utf-8")
    results = inspect_workflow(
        roles_path=ROOT / "docs/AGENT_ROLES.yaml",
        registry_path=ROOT / "docs/AGENT_REGISTRY.json",
        bindings_path=binding_path,
        tasks_db=broker.tasks_db,
        state_db=broker.state_db,
        lease_path=lease_path,
    )
    by_check = {item["check"]: item for item in results}
    assert by_check["role_keys"]["level"] == "OK"
    assert by_check["binding:agent_eval"]["level"] == "WARN"
    assert by_check["tasks_db"]["level"] == "OK"


def test_doctor_strict_binding_and_duplicate_task_checks_fail(tmp_path: Path) -> None:
    broker, binding_path, lease_path = _fixtures(tmp_path)
    lease_path.write_text(json.dumps({"root_agent": "root", "enabled": True}), encoding="utf-8")
    broker.set_desired_state("RUNNING", reason="test")
    broker.dispatch(task_id="T-one", target_agent="agent_infra", objective="one")
    first = broker.claim("agent_infra")
    assert first
    # Insert a second active task directly to exercise the doctor's read-only audit.
    with sqlite3.connect(broker.tasks_db) as db:
        db.execute(
            "INSERT INTO tasks(task_id,target_agent,objective,context_json,constraints_json,done_when_json,status,created_at,updated_at,lease_token,lease_expires_at,attempts) "
            "VALUES ('T-two','agent_infra','two','{}','{}','[]','RUNNING',0,0,'token',9999999999,1)"
        )
        db.commit()
    results = inspect_workflow(
        roles_path=ROOT / "docs/AGENT_ROLES.yaml",
        registry_path=ROOT / "docs/AGENT_REGISTRY.json",
        bindings_path=binding_path,
        tasks_db=broker.tasks_db,
        state_db=broker.state_db,
        lease_path=lease_path,
        require_all_bound=True,
        require_adapter=True,
    )
    assert any(item["check"] == "single_active_task" and item["level"] == "ERROR" for item in results)
    assert any(item["check"] == "binding:agent_eval" and item["level"] == "ERROR" for item in results)
    assert any(item["check"] == "runtime_adapter" and item["level"] == "ERROR" for item in results)
