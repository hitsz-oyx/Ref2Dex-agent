from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.agent_broker import AgentBroker
from scripts.agent_runtime_adapter import main as adapter_main


ROOT = Path(__file__).resolve().parents[1]


def _broker(tmp_path: Path) -> tuple[AgentBroker, Path]:
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
    broker.set_desired_state("RUNNING", reason="test")
    return broker, binding_path


def _fake_launcher(path: Path) -> None:
    path.write_text(
        f"import json, os, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from scripts.agent_broker import AgentBroker\n"
        "payload = json.loads(sys.stdin.readline())\n"
        "broker = AgentBroker(os.environ['REF2DEX_TASKS_DB'], os.environ['REF2DEX_STATE_DB'], "
        "os.environ['REF2DEX_ROLES'], os.environ['REF2DEX_BINDINGS'])\n"
        "broker.handoff(task_id=payload['task_id'], agent_key=os.environ['REF2DEX_AGENT_KEY'], "
        "status='COMPLETED', result={'fake_runtime': True}, evidence=['fake'], "
        "lease_token=os.environ['REF2DEX_LEASE_TOKEN'])\n",
        encoding="utf-8",
    )


def test_runtime_adapter_consumes_dispatch_and_requires_handoff(tmp_path: Path) -> None:
    broker, binding_path = _broker(tmp_path)
    broker.dispatch(task_id="T-adapter", target_agent="agent_infra", objective="run fake")
    launcher = tmp_path / "fake_launcher.py"
    _fake_launcher(launcher)

    result = adapter_main([
        "--agent", "agent_infra",
        "--command", f"{sys.executable} {launcher}",
        "--tasks-db", str(broker.tasks_db),
        "--state-db", str(broker.state_db),
        "--roles", str(ROOT / "docs/AGENT_ROLES.yaml"),
        "--bindings", str(binding_path),
        "--request-dir", str(tmp_path / "requests"),
        "--once",
    ])
    assert result == 0
    task = next(item for item in broker.status()["tasks"] if item["task_id"] == "T-adapter")
    assert task["status"] == "COMPLETED"
    request = json.loads((tmp_path / "requests/T-adapter.json").read_text(encoding="utf-8"))
    assert request["provider_request"]["type"] == "TASK_DISPATCH"


def test_runtime_adapter_marks_launcher_without_handoff_failed(tmp_path: Path) -> None:
    broker, binding_path = _broker(tmp_path)
    broker.dispatch(task_id="T-no-handoff", target_agent="agent_infra", objective="fail fake")
    launcher = tmp_path / "no_handoff.py"
    launcher.write_text("import sys; sys.stdin.read()\n", encoding="utf-8")

    result = adapter_main([
        "--agent", "agent_infra",
        "--command", f"{sys.executable} {launcher}",
        "--tasks-db", str(broker.tasks_db),
        "--state-db", str(broker.state_db),
        "--roles", str(ROOT / "docs/AGENT_ROLES.yaml"),
        "--bindings", str(binding_path),
        "--once",
    ])
    assert result == 1
    task = next(item for item in broker.status()["tasks"] if item["task_id"] == "T-no-handoff")
    assert task["status"] == "FAILED"
