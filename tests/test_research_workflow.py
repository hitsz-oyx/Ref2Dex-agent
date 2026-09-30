from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/researchctl.py"


def call(config: Path, *args: str) -> dict:
    proc = subprocess.run([sys.executable, str(CLI), "--config", str(config), *args],
                          capture_output=True, text=True, cwd=ROOT)
    return {"code": proc.returncode, **json.loads(proc.stdout)}


def setup(tmp_path: Path) -> Path:
    backend = tmp_path / "backend.py"
    backend.write_text(r'''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
store = Path(args[args.index("--orchestrator-dir") + 1])
store.mkdir(parents=True, exist_ok=True)
data = store / "fake.json"
tasks = json.loads(data.read_text()) if data.exists() else []
cmd = next(x for x in args if x in ("launch", "read", "ps", "resume"))
if cmd == "launch":
    ident = "execution-" + str(len(tasks) + 1)
    name = args[args.index("--name") + 1]
    prompt = args[-1]
    result = {"taskId": ident, "name": name, "status": "succeeded",
              "output": "delivery", "account": os.environ.get("CODEX_HOME")}
    plan = Path.cwd() / "plan.json"
    if ":root-" in name and plan.exists():
        actions = json.loads(plan.read_text())
        result["output"] = json.dumps(actions.pop(0)) if actions else json.dumps({"action": "idle", "reason": "no next task"})
        plan.write_text(json.dumps(actions))
    tasks.append(result)
    data.write_text(json.dumps(tasks))
    if (store / "crash-after-launch").exists():
        (store / "crash-after-launch").unlink()
        sys.exit(1)
    print(json.dumps(result))
elif cmd == "ps":
    print(json.dumps({"tasks": tasks}))
else:
    ident = args[args.index(cmd) + 1]
    print(json.dumps(next(t for t in tasks if t["taskId"] == ident)))
''')
    roles = {}
    for role in ("root", "agent_cm", "agent_rl", "agent_eval", "agent_infra"):
        workspace = tmp_path / role
        workspace.mkdir()
        roles[role] = {"provider": role, "runtime": "codex", "codex_home": str(tmp_path / (role + "-account")),
                       "workspace": str(workspace), "store": str(tmp_path / (role + "-store"))}
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"command": [sys.executable, str(backend)], "state_dir": str(tmp_path / "control"),
                                 "roles_file": str(ROOT / "docs/AGENT_ROLES.yaml"), "bindings": roles,
                                 "campaign": {"max_dispatches": 3, "max_root_turns": 12,
                                              "wall_time_seconds": 120, "max_gpu": 1}}))
    return config


def contract(tmp_path: Path, ident: str = "task-one", role: str = "agent_cm") -> Path:
    path = tmp_path / (ident + ".json")
    path.write_text(json.dumps({"task_id": ident, "role": role, "objective": "Distinguish two designs",
                                "decision_test": "Run the smallest engineering check", "gpu": 0,
                                "timeout_seconds": 10, "stop_conditions": ["ten seconds"],
                                "deliverables": ["result and evidence"]}))
    return path


def test_pause_stops_dispatch_and_resume_delivers_a_real_execution(tmp_path: Path) -> None:
    config = setup(tmp_path)
    task = contract(tmp_path)
    assert call(config, "supervisor", "pause")["mode"] == "paused"
    assert call(config, "dispatch", "--task", str(task))["code"] != 0
    assert call(config, "supervisor", "resume", "--legacy-dispatch-disabled")["mode"] == "active"
    launched = call(config, "dispatch", "--task", str(task))
    assert launched["code"] == 0
    assert launched["execution_id"] == "execution-1"
    status = call(config, "supervisor", "status")
    assert status["deliveries"][0]["execution"]["status"] == "succeeded"
    assert status["deliveries"][0]["acceptance"] is None


def test_roles_keep_independent_accounts_and_completed_work_drains_while_paused(tmp_path: Path) -> None:
    config = setup(tmp_path)
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    cm = contract(tmp_path)
    rl = contract(tmp_path, "task-two", "agent_rl")
    assert call(config, "dispatch", "--task", str(cm))["code"] == 0
    assert call(config, "dispatch", "--task", str(rl))["code"] == 0
    call(config, "supervisor", "pause")
    status = call(config, "supervisor", "status")
    accounts = [delivery["execution"]["account"] for delivery in status["deliveries"]]
    assert accounts == [str(tmp_path / "agent_cm-account"), str(tmp_path / "agent_rl-account")]
    assert call(config, "accept", "task-one", "--decision", "accepted", "--reason", "evidence checked")["code"] == 0
    assert call(config, "dispatch", "--task", str(contract(tmp_path, "task-three")))["code"] != 0


def test_store_identity_change_and_shared_store_are_rejected(tmp_path: Path) -> None:
    config = setup(tmp_path)
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    call(config, "dispatch", "--task", str(contract(tmp_path)))
    document = json.loads(config.read_text())
    document["bindings"]["agent_cm"]["codex_home"] = str(tmp_path / "different-account")
    config.write_text(json.dumps(document))
    assert "identity changed" in call(config, "supervisor", "status")["error"]
    document["bindings"]["agent_rl"]["store"] = document["bindings"]["agent_cm"]["store"]
    config.write_text(json.dumps(document))
    assert "independent" in call(config, "supervisor", "status")["error"]


def test_contract_reuse_and_resource_limits(tmp_path: Path) -> None:
    config = setup(tmp_path)
    task = contract(tmp_path)
    assert call(config, "supervisor", "resume")["code"] != 0
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    first = call(config, "dispatch", "--task", str(task))
    second = call(config, "dispatch", "--task", str(task))
    assert first["execution_id"] == second["execution_id"]
    assert second["reused"] is True
    excessive = json.loads(task.read_text())
    excessive["task_id"] = "gpu-task"
    excessive["gpu"] = 1
    task.write_text(json.dumps(excessive))
    assert "GPU" in call(config, "dispatch", "--task", str(task))["error"]


def plan(config: Path, actions: list) -> None:
    directory = Path(json.loads(config.read_text())["bindings"]["root"]["workspace"])
    (directory / "plan.json").write_text(json.dumps(actions))


def test_background_supervisor_accepts_and_dispatches_next_task(tmp_path: Path) -> None:
    config = setup(tmp_path)
    first = json.loads(contract(tmp_path).read_text())
    second = json.loads(contract(tmp_path, "task-two").read_text())
    plan(config, [{"action": "dispatch", "task": first},
                  {"action": "accept", "task_id": "task-one", "reason": "matched evidence checked"},
                  {"action": "dispatch", "task": second},
                  {"action": "idle", "reason": "await user"}])
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    for _ in range(8):
        assert call(config, "supervisor", "tick")["code"] == 0
    status = call(config, "supervisor", "status")
    workers = [item for item in status["deliveries"] if item["role"] != "root"]
    assert [item["task_id"] for item in workers] == ["task-one", "task-two"]
    assert workers[0]["acceptance"]["decision"] == "accepted"
    before = len(status["deliveries"])
    for _ in range(3):
        assert call(config, "supervisor", "tick")["state"] == "idle"
    assert len(call(config, "supervisor", "status")["deliveries"]) == before


def test_lost_launch_response_is_reconciled_without_duplicate_execution(tmp_path: Path) -> None:
    config = setup(tmp_path)
    data = json.loads(config.read_text())
    store = Path(data["bindings"]["agent_cm"]["store"])
    task = contract(tmp_path)
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    (store / "crash-after-launch").touch()
    assert call(config, "dispatch", "--task", str(task))["code"] != 0
    assert call(config, "dispatch", "--task", str(task))["code"] != 0
    assert call(config, "reconcile")["unresolved"] == []
    assert call(config, "dispatch", "--task", str(task))["execution_id"] == "execution-1"
    assert len(call(config, "supervisor", "status")["deliveries"]) == 1


def test_recovery_is_finite_and_pause_preserves_pending_root_decision(tmp_path: Path) -> None:
    import time
    config = setup(tmp_path)
    document = json.loads(config.read_text())
    document["max_recovery_attempts"] = 2
    document["recovery_backoff_seconds"] = 0.01
    config.write_text(json.dumps(document))
    task = json.loads(contract(tmp_path).read_text())
    plan(config, [{"action": "dispatch", "task": task}])
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    assert call(config, "supervisor", "tick")["state"] == "root_started"
    call(config, "supervisor", "pause")
    assert call(config, "supervisor", "tick")["state"] == "paused"
    assert len(call(config, "supervisor", "status")["deliveries"]) == 1
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    assert call(config, "supervisor", "tick")["action"] == "dispatch"
    call(config, "accept", "task-one", "--decision", "accepted", "--reason", "evidence checked")
    plan(config, [{"action": "unsupported"}, {"action": "unsupported"}])
    for _ in range(2):
        assert call(config, "supervisor", "tick")["state"] == "root_started"
        call(config, "supervisor", "tick")
        time.sleep(0.03)
    assert call(config, "supervisor", "status")["mode"] == "attention"
    assert call(config, "supervisor", "tick")["state"] == "attention"


def test_unknown_launch_does_not_retry_and_budget_is_not_reset_by_resume(tmp_path: Path) -> None:
    config = setup(tmp_path)
    document = json.loads(config.read_text())
    document["campaign"]["max_dispatches"] = 1
    config.write_text(json.dumps(document))
    task = contract(tmp_path)
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    assert call(config, "dispatch", "--task", str(task))["code"] == 0
    call(config, "accept", "task-one", "--decision", "accepted", "--reason", "checked")
    call(config, "supervisor", "pause")
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    assert "budget" in call(config, "dispatch", "--task", str(contract(tmp_path, "task-two")))["error"]


def test_frontend_cannot_dispatch_while_background_owner_is_running(tmp_path: Path) -> None:
    import time
    config = setup(tmp_path)
    document = json.loads(config.read_text())
    document["poll_seconds"] = 0.05
    config.write_text(json.dumps(document))
    # Run while paused: the background owner observes and never launches root.
    proc = subprocess.Popen([sys.executable, str(CLI), "--config", str(config), "supervisor", "run"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for _ in range(30):
            if call(config, "supervisor", "status")["control"]["pid"] == proc.pid:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("supervisor did not acquire ownership")
        assert "another supervisor" in call(config, "dispatch", "--task", str(contract(tmp_path)))["error"]
        assert call(config, "supervisor", "run", "--detach")["code"] != 0
        assert call(config, "supervisor", "pause")["mode"] == "paused"
    finally:
        proc.terminate()
        proc.communicate(timeout=5)
    assert call(config, "supervisor", "tick")["state"] == "paused"


def test_pause_remains_available_after_a_binding_identity_mismatch(tmp_path: Path) -> None:
    config = setup(tmp_path)
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    document = json.loads(config.read_text())
    document["bindings"]["root"]["codex_home"] = str(tmp_path / "another-account")
    config.write_text(json.dumps(document))
    assert call(config, "supervisor", "status")["code"] != 0
    assert call(config, "supervisor", "pause")["mode"] == "paused"


def test_detached_supervisor_survives_frontend_exit_and_recovers_its_child(tmp_path: Path) -> None:
    import os
    import signal
    import time
    config = setup(tmp_path)
    document = json.loads(config.read_text())
    document.update(poll_seconds=0.03, recovery_backoff_seconds=0.03)
    config.write_text(json.dumps(document))
    task = json.loads(contract(tmp_path).read_text())
    plan(config, [{"action": "dispatch", "task": task}, {"action": "idle", "reason": "wait"}])
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    guardian = call(config, "supervisor", "run", "--detach")["pid"]
    child = None
    try:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            status = call(config, "supervisor", "status")
            child = status["control"]["pid"]
            if child and any(item["role"] == "agent_cm" for item in status["deliveries"]):
                break
            time.sleep(0.03)
        else:
            raise AssertionError("detached supervisor did not dispatch after frontend exit")
        old_child = child
        os.kill(child, signal.SIGKILL)
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            status = call(config, "supervisor", "status")
            child = status["control"]["pid"]
            if child and child != old_child:
                break
            time.sleep(0.03)
        else:
            raise AssertionError("guardian did not recover its own child")
        workers = [item for item in status["deliveries"] if item["role"] != "root"]
        assert [item["task_id"] for item in workers] == ["task-one"]
        call(config, "supervisor", "pause")
    finally:
        call(config, "supervisor", "pause")
        for pid in (guardian, child):
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
