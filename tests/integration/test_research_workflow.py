from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
    if (store / "provider-down").exists():
        result["status"] = "failed"
        result["error"] = "quota exhausted"
    if (store / "provider-error").exists():
        result["status"] = "failed"
        result["error"] = (store / "provider-error").read_text()
    tasks.append(result)
    data.write_text(json.dumps(tasks))
    if (store / "crash-after-launch").exists():
        (store / "crash-after-launch").unlink()
        sys.exit(1)
    print(json.dumps(result))
elif cmd == "ps":
    print(json.dumps({"tasks": tasks}))
else:
    if (store / "delay-read").exists():
        import time
        (store / "reading").touch()
        time.sleep(0.5)
    ident = args[args.index(cmd) + 1]
    print(json.dumps(next(t for t in tasks if t["taskId"] == ident)))
''')
    roles = {}
    for role in ("root", "agent_cm", "agent_rl", "agent_eval", "agent_infra"):
        workspace = tmp_path / role
        workspace.mkdir()
        subprocess.run(["git", "init", "-q", str(workspace)], check=True)
        (workspace / ".git/info/exclude").write_text("plan.json\ndocs/\n")
        roles[role] = {"provider": role, "runtime": "codex", "codex_home": str(tmp_path / (role + "-account")),
                       "workspace": str(workspace), "store": str(tmp_path / (role + "-store"))}
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"command": [sys.executable, str(backend)], "state_dir": str(tmp_path / "control"),
                                 "roles_file": str(ROOT / "docs/AGENT_ROLES.yaml"), "bindings": roles,
                                 "campaign": {"max_dispatches": 3, "max_root_turns": 12,
                                              "wall_time_seconds": 120, "max_gpu": 1, "allowed_workspace_roots": [str(tmp_path)]}}))
    return config


def contract(tmp_path: Path, ident: str = "task-one", role: str = "agent_cm") -> Path:
    path = tmp_path / (ident + ".json")
    path.write_text(json.dumps({"task_id": ident, "role": role, "kind": "engineering", "objective": "Distinguish two designs",
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


def test_recovery_is_finite_and_pause_requires_fresh_root_decision(tmp_path: Path) -> None:
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
    assert call(config, "supervisor", "tick")["state"] == "stale_decision"
    plan(config, [{"action": "dispatch", "task": task}])
    assert call(config, "supervisor", "tick")["state"] == "root_started"
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
            if child and status["control"]["idle_fingerprint"] is not None and any(item["role"] == "agent_cm" for item in status["deliveries"]):
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


def test_account_refresh_is_allowed_but_account_and_implicit_config_changes_are_sealed(tmp_path: Path) -> None:
    config = setup(tmp_path)
    account = tmp_path / "root-account"
    account.mkdir()
    auth = account / "auth.json"
    auth.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"account_id": "account-A", "access_token": "old"}}))
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    auth.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"account_id": "account-A", "access_token": "refreshed"}}))
    assert call(config, "supervisor", "status")["code"] == 0
    auth.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"account_id": "account-B", "access_token": "new"}}))
    assert "identity changed" in call(config, "supervisor", "status")["error"]
    auth.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"account_id": "account-A", "access_token": "refreshed"}}))
    (tmp_path / "root" / "orchestrator.config.json").write_text('{"agents":{}}')
    assert "identity changed" in call(config, "supervisor", "status")["error"]


def test_experiment_permissions_and_workspace_roots_are_enforced(tmp_path: Path) -> None:
    config = setup(tmp_path)
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    path = contract(tmp_path, "infra-experiment", "agent_infra")
    value = json.loads(path.read_text())
    value["kind"] = "probe"
    path.write_text(json.dumps(value))
    assert "not authorized" in call(config, "dispatch", "--task", str(path))["error"]
    document = json.loads(config.read_text())
    document["campaign"]["allowed_workspace_roots"] = [str(tmp_path / "agent_cm")]
    config.write_text(json.dumps(document))
    assert "outside authorized" in call(config, "supervisor", "status")["error"]


def test_completed_history_read_failure_does_not_block_new_research(tmp_path: Path) -> None:
    config = setup(tmp_path)
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    call(config, "dispatch", "--task", str(contract(tmp_path)))
    call(config, "accept", "task-one", "--decision", "accepted", "--reason", "evidence verified")
    # The external executor loses an old result after acceptance. Current work
    # should still be scheduled; the public historical read remains unknown.
    document = json.loads(config.read_text())
    store = Path(document["bindings"]["agent_cm"]["store"])
    (store / "fake.json").write_text('[]')
    plan(config, [{"action": "idle", "reason": "waiting"}])
    assert call(config, "supervisor", "tick")["state"] == "root_started"
    assert call(config, "supervisor", "tick")["action"] == "idle"


def test_pause_during_a_slow_backend_read_preserves_then_invalidates_old_decision(tmp_path: Path) -> None:
    import time
    config = setup(tmp_path)
    task = json.loads(contract(tmp_path).read_text())
    plan(config, [{"action": "dispatch", "task": task}])
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    assert call(config, "supervisor", "tick")["state"] == "root_started"
    store = Path(json.loads(config.read_text())["bindings"]["root"]["store"])
    (store / "delay-read").touch()
    proc = subprocess.Popen([sys.executable, str(CLI), "--config", str(config), "supervisor", "tick"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (store / "reading").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (store / "reading").exists()
        assert call(config, "supervisor", "pause")["mode"] == "paused"
        stdout, stderr = proc.communicate(timeout=5)
        assert proc.returncode == 0, stderr
        assert json.loads(stdout)["state"] == "paused"
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.communicate(timeout=5)
        (store / "delay-read").unlink()
    status = call(config, "supervisor", "status")
    assert len(status["deliveries"]) == 1
    assert status["deliveries"][0]["applied"] is False
    call(config, "supervisor", "resume", "--legacy-dispatch-disabled")
    assert call(config, "supervisor", "tick")["state"] == "stale_decision"
    plan(config, [{"action": "dispatch", "task": task}])
    assert call(config, "supervisor", "tick")["state"] == "root_started"
    assert call(config, "supervisor", "tick")["action"] == "dispatch"


def test_unverified_model_harness_is_rejected(tmp_path: Path) -> None:
    config = setup(tmp_path)
    document = json.loads(config.read_text())
    binding = document['bindings']['agent_cm']
    binding['runtime'] = 'claude-code'
    config.write_text(json.dumps(document))
    assert 'account isolation is unverified' in call(config, 'supervisor', 'status')['error']
    binding['engineering_only'] = True
    binding['runtime_config'] = str(tmp_path / 'unused.json')
    config.write_text(json.dumps(document))
    assert 'account isolation is unverified' in call(config, 'supervisor', 'status')['error']
    binding['runtime'] = 'codex-other'
    config.write_text(json.dumps(document))
    assert call(config, 'supervisor', 'status')['code'] != 0
    binding['runtime'] = 'engineering-process'
    binding['engineering_only'] = True
    runtime_config = tmp_path / 'custom.json'
    runtime_config.write_text(json.dumps({'agents': {'engineering-process': {'adapter': 'process', 'command': sys.executable}}}))
    binding['runtime_config'] = str(runtime_config)
    config.write_text(json.dumps(document))
    assert call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')['code'] == 0
    task = contract(tmp_path)
    value = json.loads(task.read_text())
    value['kind'] = 'probe'
    task.write_text(json.dumps(value))
    assert 'engineering-only' in call(config, 'dispatch', '--task', str(task))['error']


def test_continuous_research_has_no_implicit_total_deadline(tmp_path: Path) -> None:
    config = setup(tmp_path)
    document = json.loads(config.read_text())
    for key in ('max_dispatches', 'max_root_turns', 'wall_time_seconds'):
        document['campaign'].pop(key)
    config.write_text(json.dumps(document))
    assert call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')['code'] == 0
    for index in range(4):
        task_id = 'continuous-' + str(index)
        assert call(config, 'dispatch', '--task', str(contract(tmp_path, task_id)))['code'] == 0
        assert call(config, 'accept', task_id, '--decision', 'accepted', '--reason', 'checked')['code'] == 0
    plan(config, [{'action': 'idle', 'reason': 'waiting for evidence'}])
    assert call(config, 'supervisor', 'tick')['state'] == 'root_started'


def test_user_instruction_is_durable_and_obsoletes_old_root_decisions(tmp_path: Path) -> None:
    config = setup(tmp_path)
    task = json.loads(contract(tmp_path).read_text())
    plan(config, [{'action': 'dispatch', 'task': task}, {'action': 'idle', 'reason': 'new instruction'}])
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    assert call(config, 'supervisor', 'tick')['state'] == 'root_started'
    submitted = call(config, 'instruct', '--request-id', 'user-1', '--text', 'Do CPU analysis first; leave independent work alone.')
    assert submitted['code'] == 0
    assert call(config, 'instruct', '--request-id', 'user-1', '--text', 'Do CPU analysis first; leave independent work alone.')['reused'] is True
    assert call(config, 'supervisor', 'tick')['state'] == 'stale_decision'
    assert call(config, 'supervisor', 'tick')['state'] == 'root_started'
    assert call(config, 'supervisor', 'tick')['action'] == 'idle'
    status = call(config, 'supervisor', 'status')
    assert not any(item['role'] == 'agent_cm' for item in status['deliveries'])
    assert status['control']['processed_version'] == status['control']['instruction_version']
    assert status['instructions'][-1]['text'].startswith('Do CPU')


def test_research_blocked_has_review_record_and_wakes_on_new_instruction(tmp_path: Path) -> None:
    config = setup(tmp_path)
    plan(config, [{'action': 'blocked', 'reason': 'No authorized discriminating direction remains.',
                  'review': 'Checked a higher-level representation route and objective alternative; both need unavailable data.',
                  'alternatives': ['representation change', 'objective change'],
                  'resume_when': 'new data or user changes the goal'}])
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    assert call(config, 'supervisor', 'tick')['state'] == 'root_started'
    applied = call(config, 'supervisor', 'tick')
    assert applied['action'] == 'blocked'
    status = call(config, 'supervisor', 'status')
    assert status['mode'] == 'blocked'
    assert status['control']['completion'] is None
    assert 'higher-level' in status['decisions'][-1]['review']
    log = Path(status['decision_record'])
    assert 'No authorized' in log.read_text()
    assert call(config, 'supervisor', 'tick')['state'] == 'blocked'
    assert call(config, 'instruct', '--request-id', 'new-data', '--text', 'New data is now available')['code'] == 0
    assert call(config, 'supervisor', 'status')['mode'] == 'active'


def test_engineering_delivery_cannot_complete_research_mission(tmp_path: Path) -> None:
    config = setup(tmp_path)
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    call(config, 'dispatch', '--task', str(contract(tmp_path)))
    call(config, 'accept', 'task-one', '--decision', 'accepted', '--reason', 'smoke passed')
    plan(config, [{'action': 'complete', 'reason': 'process succeeded', 'evidence_task_ids': ['task-one']}])
    call(config, 'supervisor', 'tick')
    rejected = call(config, 'supervisor', 'tick')
    assert 'formal Validation' in rejected['error']
    assert call(config, 'supervisor', 'status')['control']['completion'] is None


def test_failed_provider_uses_only_verified_backup_and_preserves_execution_binding(tmp_path: Path) -> None:
    config = setup(tmp_path)
    document = json.loads(config.read_text())
    primary = document['bindings']['agent_cm']
    backup = {**primary, 'provider': 'backup-cm', 'codex_home': str(tmp_path / 'backup-account'),
              'store': str(tmp_path / 'backup-store'), 'verified': True}
    primary['fallbacks'] = [backup]
    config.write_text(json.dumps(document))
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    (Path(primary['store']) / 'provider-down').touch()
    first = call(config, 'dispatch', '--task', str(contract(tmp_path)))
    assert first['code'] == 0
    assert call(config, 'accept', 'task-one', '--decision', 'rejected', '--reason', 'quota exhausted')['code'] == 0
    second = call(config, 'dispatch', '--task', str(contract(tmp_path, 'task-two')))
    assert second['code'] == 0
    status = call(config, 'supervisor', 'status')
    assert status['deliveries'][0]['execution']['account'] == primary['codex_home']
    assert status['deliveries'][1]['execution']['account'] == backup['codex_home']
    assert status['deliveries'][0]['execution_id'] == first['execution_id']
    assert status['decisions'][-1]['action'] == 'provider_switch'


def test_requested_provider_waits_without_switching_or_launching_again(tmp_path: Path) -> None:
    config = setup(tmp_path)
    document = json.loads(config.read_text())
    primary = document['bindings']['agent_cm']
    backup = {**primary, 'provider': 'backup-cm', 'codex_home': str(tmp_path / 'backup-account'),
              'store': str(tmp_path / 'backup-store'), 'verified': True}
    primary['fallbacks'] = [backup]
    config.write_text(json.dumps(document))
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    (Path(primary['store']) / 'provider-down').touch()
    call(config, 'dispatch', '--task', str(contract(tmp_path)))
    call(config, 'accept', 'task-one', '--decision', 'rejected', '--reason', 'quota exhausted')
    next_task = contract(tmp_path, 'task-two')
    value = json.loads(next_task.read_text())
    value['provider'] = 'agent_cm'
    next_task.write_text(json.dumps(value))
    assert 'no available permitted provider' in call(config, 'dispatch', '--task', str(next_task))['error']
    assert len(call(config, 'supervisor', 'status')['deliveries']) == 1
    (Path(primary['store']) / 'provider-down').unlink()
    assert call(config, 'provider-ready', '--role', 'agent_cm', '--provider', 'agent_cm')['code'] == 0
    assert call(config, 'dispatch', '--task', str(next_task))['code'] == 0
    assert call(config, 'supervisor', 'status')['deliveries'][-1]['execution']['account'] == primary['codex_home']


def test_independent_changes_are_left_untouched_and_not_dispatched_into(tmp_path: Path) -> None:
    config = setup(tmp_path)
    directory = Path(json.loads(config.read_text())['bindings']['agent_cm']['workspace'])
    subprocess.run(['git', 'init', '-q', str(directory)], check=True)
    independent = directory / 'user-work.txt'
    independent.write_text('independent unfinished work\n')
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    result = call(config, 'dispatch', '--task', str(contract(tmp_path)))
    assert 'independent or uncommitted work' in result['error']
    assert independent.read_text() == 'independent unfinished work\n'
    assert call(config, 'supervisor', 'status')['deliveries'] == []


def test_formal_eval_can_complete_current_goal_but_old_goal_cannot(tmp_path: Path) -> None:
    config = setup(tmp_path)
    document = json.loads(config.read_text())
    document['campaign']['max_dispatches'] = 10
    config.write_text(json.dumps(document))
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    for ident, role in [('validation-source', 'agent_rl'), ('eval-review', 'agent_eval')]:
        path = contract(tmp_path, ident, role)
        value = json.loads(path.read_text())
        value['kind'] = 'validation'
        path.write_text(json.dumps(value))
        assert call(config, 'dispatch', '--task', str(path))['code'] == 0
        assert call(config, 'accept', ident, '--decision', 'accepted', '--reason', 'formal evidence checked')['code'] == 0
    workspace = Path(document['bindings']['root']['workspace'])
    cards = workspace / 'docs/experiments/validations'
    cards.mkdir(parents=True)
    (cards / 'VAL-baseline.md').write_text('Validation: VAL-baseline\nSUPPORTED\n')
    (cards / 'VAL-cm.md').write_text('Validation: VAL-cm\nSUPPORTED\n')
    report = {'schema': 'ref2dex.validation-evidence.v1', 'goal_version': 0,
              'reviewed_task_ids': ['validation-source'], 'checks': {
                  'self_trained_grasp': {'verdict': 'SUPPORTED', 'validation_id': 'VAL-baseline',
                     'card': 'docs/experiments/validations/VAL-baseline.md', 'scope': 'fixed grasp task',
                     'pre_registered': True, 'seeds': [1, 2, 3], 'self_trained': True},
                  'cm_policy_utility': {'verdict': 'SUPPORTED', 'validation_id': 'VAL-cm',
                     'card': 'docs/experiments/validations/VAL-cm.md', 'scope': 'fixed Cm task',
                     'pre_registered': True, 'seeds': [1, 2, 3], 'matched_control': True, 'arms': ['Cm-on', 'Cm-off']}}}
    data = Path(document['bindings']['agent_eval']['store']) / 'fake.json'
    tasks = json.loads(data.read_text())
    tasks[0]['output'] = json.dumps(report)
    data.write_text(json.dumps(tasks))
    plan(config, [{'action': 'complete', 'reason': 'both MISSION objectives independently reviewed',
                   'evidence_task_ids': ['validation-source', 'eval-review']}])
    assert call(config, 'supervisor', 'tick')['state'] == 'root_started'
    assert call(config, 'supervisor', 'tick')['action'] == 'complete'
    assert call(config, 'supervisor', 'status')['mode'] == 'completed'
    assert call(config, 'supervisor', 'tick')['state'] == 'completed'
    # A new goal cannot reuse accepted evidence as current-goal completion.
    call(config, 'instruct', '--request-id', 'new-goal', '--text', 'Expand the task scope', '--new-goal')
    assert call(config, 'supervisor', 'status')['mode'] == 'paused'
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    plan(config, [{'action': 'complete', 'reason': 'reuse prior evidence',
                   'evidence_task_ids': ['validation-source', 'eval-review']}])
    call(config, 'supervisor', 'tick')
    assert 'current-goal' in call(config, 'supervisor', 'tick')['error']


def test_changed_instruction_waits_for_running_old_root_before_starting_another(tmp_path: Path) -> None:
    config = setup(tmp_path)
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    plan(config, [{'action': 'idle', 'reason': 'old goal'}])
    call(config, 'supervisor', 'tick')
    store = Path(json.loads(config.read_text())['bindings']['root']['store']) / 'fake.json'
    executions = json.loads(store.read_text())
    executions[0]['status'] = 'running'
    store.write_text(json.dumps(executions))
    call(config, 'instruct', '--request-id', 'changed', '--text', 'Adjust the goal')
    assert call(config, 'supervisor', 'tick')['state'] == 'waiting_stale_root'
    assert len(call(config, 'supervisor', 'status')['deliveries']) == 1
    executions[0]['status'] = 'succeeded'
    store.write_text(json.dumps(executions))
    assert call(config, 'supervisor', 'tick')['state'] == 'stale_decision'


def test_idle_decision_cannot_hide_worker_completion_during_root_turn(tmp_path: Path) -> None:
    config = setup(tmp_path)
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    call(config, 'dispatch', '--task', str(contract(tmp_path)))
    worker_store = Path(json.loads(config.read_text())['bindings']['agent_cm']['store']) / 'fake.json'
    worker = json.loads(worker_store.read_text())
    worker[0]['status'] = 'running'
    worker_store.write_text(json.dumps(worker))
    plan(config, [{'action': 'idle', 'reason': 'worker was running at observation time'},
                  {'action': 'accept', 'task_id': 'task-one', 'reason': 'finished evidence checked'}])
    assert call(config, 'supervisor', 'tick')['state'] == 'root_started'
    worker[0]['status'] = 'succeeded'
    worker_store.write_text(json.dumps(worker))
    assert call(config, 'supervisor', 'tick')['action'] == 'idle'
    assert call(config, 'supervisor', 'tick')['state'] == 'root_started'
    assert call(config, 'supervisor', 'tick')['action'] == 'accept'


def test_stop_background_owner_preserves_worker_results(tmp_path: Path) -> None:
    import time
    config = setup(tmp_path)
    document = json.loads(config.read_text())
    document['poll_seconds'] = 0.05
    config.write_text(json.dumps(document))
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    call(config, 'dispatch', '--task', str(contract(tmp_path)))
    call(config, 'supervisor', 'pause')
    proc = subprocess.Popen([sys.executable, str(CLI), '--config', str(config), 'supervisor', 'run'], stdout=subprocess.PIPE)
    try:
        for _ in range(20):
            if call(config, 'supervisor', 'status')['supervisor_running']:
                break
            time.sleep(0.05)
        assert call(config, 'supervisor', 'stop')['state'] == 'stop_requested'
        assert proc.wait(timeout=5) == 0
        status = call(config, 'supervisor', 'status')
        assert status['mode'] == 'paused'
        assert status['supervisor_running'] is False
        assert status['deliveries'][0]['execution']['status'] == 'succeeded'
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_shared_provider_account_directory_is_rejected(tmp_path: Path) -> None:
    config = setup(tmp_path)
    document = json.loads(config.read_text())
    primary = document['bindings']['agent_cm']
    primary['fallbacks'] = [{**primary, 'provider': 'backup', 'store': str(tmp_path / 'backup-store'), 'verified': True}]
    config.write_text(json.dumps(document))
    assert 'independent CODEX_HOME' in call(config, 'supervisor', 'status')['error']


def test_engineering_fallback_cannot_run_research_or_supervise_root(tmp_path: Path) -> None:
    config = setup(tmp_path)
    definition = tmp_path / 'runtime.json'
    definition.write_text(json.dumps({'agents': {'engineering': {'adapter': 'process', 'command': sys.executable}}}))
    document = json.loads(config.read_text())
    primary = document['bindings']['agent_cm']
    primary['fallbacks'] = [{'provider': 'engineering', 'runtime': 'engineering', 'engineering_only': True,
        'runtime_config': str(definition), 'store': str(tmp_path / 'engineering-store'),
        'workspace': primary['workspace'], 'verified': True}]
    config.write_text(json.dumps(document))
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    task = contract(tmp_path)
    value = json.loads(task.read_text());value.update(kind='probe', provider='engineering')
    task.write_text(json.dumps(value))
    assert 'engineering-only' in call(config, 'dispatch', '--task', str(task))['error']
    document['bindings']['root']['fallbacks'] = [{**primary['fallbacks'][0], 'workspace': document['bindings']['root']['workspace']}]
    config.write_text(json.dumps(document))
    assert 'root bindings require Codex' in call(config, 'supervisor', 'status')['error']


def test_non_git_workspace_is_not_adopted(tmp_path: Path) -> None:
    import shutil
    config = setup(tmp_path)
    document = json.loads(config.read_text())
    shutil.rmtree(Path(document['bindings']['agent_cm']['workspace']) / '.git')
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    assert 'own Git worktree' in call(config, 'dispatch', '--task', str(contract(tmp_path)))['error']
    assert call(config, 'supervisor', 'status')['deliveries'] == []


def test_provider_readiness_wins_over_old_inflight_failure_observation(tmp_path: Path) -> None:
    import time
    config = setup(tmp_path)
    document = json.loads(config.read_text())
    primary = document['bindings']['agent_cm']
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    store = Path(primary['store'])
    (store / 'provider-down').touch()
    call(config, 'dispatch', '--task', str(contract(tmp_path)))
    call(config, 'accept', 'task-one', '--decision', 'rejected', '--reason', 'quota exhausted')
    (store / 'delay-read').touch()
    task = contract(tmp_path, 'task-two')
    proc = subprocess.Popen([sys.executable, str(CLI), '--config', str(config), 'dispatch', '--task', str(task)], stdout=subprocess.PIPE)
    try:
        for _ in range(100):
            if (store / 'reading').exists(): break
            time.sleep(0.01)
        assert (store / 'reading').exists()
        (store / 'provider-down').unlink()
        assert call(config, 'provider-ready', '--role', 'agent_cm', '--provider', 'agent_cm')['code'] == 0
        output, _ = proc.communicate(timeout=10)
        assert proc.returncode == 0, output.decode()
        assert call(config, 'supervisor', 'status')['control']['provider_waits'] == {}
    finally:
        if proc.poll() is None: proc.terminate();proc.wait(timeout=5)


def test_confirmed_connection_failures_switch_after_finite_attempts(tmp_path: Path) -> None:
    config = setup(tmp_path)
    document = json.loads(config.read_text());document['max_recovery_attempts'] = 2
    primary = document['bindings']['agent_cm']
    primary['fallbacks'] = [{**primary, 'provider': 'backup', 'codex_home': str(tmp_path / 'backup-account'),
        'store': str(tmp_path / 'backup-store'), 'verified': True}]
    config.write_text(json.dumps(document))
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    store = Path(primary['store'])
    (store / 'provider-error').write_text('provider connection refused')
    for ident in ('task-one', 'task-two'):
        assert call(config, 'dispatch', '--task', str(contract(tmp_path, ident)))['code'] == 0
        assert call(config, 'accept', ident, '--decision', 'rejected', '--reason', 'provider connection failed')['code'] == 0
    assert call(config, 'dispatch', '--task', str(contract(tmp_path, 'task-three')))['code'] == 0
    deliveries = call(config, 'supervisor', 'status')['deliveries']
    assert [d['binding_store'] for d in deliveries] == [primary['store'], primary['store'], str(tmp_path / 'backup-store')]


def test_major_decision_result_can_update_same_record_and_shows_control_scope(tmp_path: Path) -> None:
    config = setup(tmp_path)
    call(config, 'instruct', '--request-id', 'scope', '--text', 'Test current authorized research scope', '--new-goal')
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    first = {'decision_id': 'route-choice', 'decision': 'Switch representation', 'evidence': 'evidence/card',
        'reason': 'Exclude failed route', 'cost_and_stop': 'one bounded test', 'outcome': 'pending'}
    plan(config, [{'action': 'idle', 'reason': 'waiting for work', 'major_decision': first}])
    call(config, 'supervisor', 'tick');call(config, 'supervisor', 'tick')
    call(config, 'instruct', '--request-id', 'result', '--text', 'The route test resolved the blocker')
    plan(config, [{'action': 'idle', 'reason': 'external wait', 'major_decision': {**first, 'outcome': 'blocker resolved'}}])
    call(config, 'supervisor', 'tick');call(config, 'supervisor', 'tick')
    call(config, 'supervisor', 'pause')
    status = call(config, 'supervisor', 'status')
    records = [d for d in status['decisions'] if d.get('decision_id') == 'route-choice']
    assert len(records) == 1
    assert records[0]['outcome'] == 'blocker resolved'
    rendered = Path(status['decision_record']).read_text()
    assert 'paused' in rendered and 'Test current authorized research scope' in rendered


def test_background_root_provider_recovers_on_verified_backup(tmp_path: Path) -> None:
    import time
    config = setup(tmp_path)
    document = json.loads(config.read_text())
    document.update(max_recovery_attempts=2, recovery_backoff_seconds=0.01)
    primary = document['bindings']['root']
    primary['fallbacks'] = [{**primary, 'provider': 'root-backup', 'codex_home': str(tmp_path / 'backup-root-account'),
        'store': str(tmp_path / 'backup-root-store'), 'verified': True}]
    config.write_text(json.dumps(document))
    call(config, 'supervisor', 'resume', '--legacy-dispatch-disabled')
    (Path(primary['store']) / 'provider-error').write_text('provider connection refused')
    for _ in range(6):
        call(config, 'supervisor', 'tick')
        time.sleep(0.03)
    status = call(config, 'supervisor', 'status')
    assert status['mode'] == 'active'
    assert [d['binding_store'] for d in status['deliveries']] == [primary['store'], primary['store'], str(tmp_path / 'backup-root-store')]
