#!/usr/bin/env python3
"""CPU-only wiring/preflight for the six-expert support contract.

This command validates the main-worktree contract on deterministic synthetic
rows, audits the real legacy evaluator source, and runs its import-safe
``--dry-run`` path.  It never imports Isaac Gym, starts a collector, queries a
GPU, writes output artifacts, or treats synthetic rows as research data.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PINNED_ROOT = Path(os.environ.get(
    "REF2DEX_MAIN_ROOT", "/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent")).resolve()
EVALUATOR = ROOT / "third_party/DExplore/dexplore/evaluate_temporal_expert_option.py"
ROUTER_EVALUATOR = ROOT / "third_party/DExplore/dexplore/evaluate_object_router.py"
COLLECTOR_CONFIG = ROOT / "src/task/CmResidual/configs/airplane_temporal_expert_probe.json"
ROUTE_CONFIG = ROOT / "src/task/CmResidual/configs/hf02_temporal_canonical_route.json"
ROUTER_MODEL = PINNED_ROOT / "outputs/CmResidual/agent_six_expert_router_model_20260925/router.joblib"

sys.path.insert(0, str(ROOT))
from src.task.CmResidual import six_expert_support_adapter as adapter  # noqa: E402


def _hashes() -> dict[str, str]:
    return {
        "route_config_sha256": "1" * 64,
        "collector_config_sha256": "2" * 64,
        "router_model_sha256": "3" * 64,
    }


def _synthetic_rows() -> list[dict[str, Any]]:
    """Create contract fixtures only; callers must not persist or use them."""
    hashes = _hashes()
    experts = [
        {"name": f"expert_{index}", "checkpoint_sha256": str(index + 4) * 64}
        for index in range(6)
    ]
    rows: list[dict[str, Any]] = []
    for split_index, split in enumerate(("fit", "holdout")):
        for ordinal in range(30):
            for arm in range(6):
                value = float(split_index * 1000 + ordinal * 6 + arm)
                pose_t = [value, 0.1, -0.2]
                pose_next = [value + 0.01, 0.08, -0.17]
                candidates = [
                    [float(candidate + component / 1000.0)
                     for component in range(18)]
                    for candidate in range(6)
                ]
                teacher_id = (arm + 1) % 6
                rows.append({
                    "episode_id": f"{split}-cpu-fixture-{ordinal:02d}-{arm}",
                    "split": split,
                    "pre_action_observation": [0.0] * 1442,
                    "object_pose_t_object_local_frame": pose_t,
                    "candidate_actions": candidates,
                    "candidate_expert_names_and_checkpoint_sha256": experts,
                    "assignment": arm,
                    "assignment_propensity": 1 / 6,
                    "executed_action": list(candidates[arm]),
                    "object_pose_t_plus_1_object_local_frame": pose_next,
                    "target_delta_object_local_1": [0.01, -0.02, 0.03],
                    "contact_mask_t_plus_1_to_t_plus_5": [True, False, True, True, False],
                    "router_teacher_candidate_id": teacher_id,
                    "router_teacher_action": list(candidates[teacher_id]),
                    "router_model_sha256": hashes["router_model_sha256"],
                    "router_input_state_sha256": "4" * 64,
                    "router_teacher_source": "c1_observation_router",
                    "start_frame": ordinal,
                    "motion_name": "airplane_lift",
                    "route_config_sha256": hashes["route_config_sha256"],
                    "collector_config_sha256": hashes["collector_config_sha256"],
                })
    return rows


def _check(name: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    result = {"name": name, "status": status, "detail": detail}
    result.update(extra)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_path(relative: str | Path) -> Path:
    value = Path(relative)
    local = (ROOT / value).resolve()
    if local.is_file():
        return local
    return (PINNED_ROOT / value).resolve()


def _dry_run() -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    env["REF2DEX_MAIN_ROOT"] = str(PINNED_ROOT)
    command = [
        sys.executable, str(EVALUATOR), "--dry-run", "--split", "fit",
        "--skip-artifact-hashes",
    ]
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True,
                               capture_output=True, check=False)
    line = next((line for line in completed.stdout.splitlines()
                 if line.startswith("REF2DEX_TEMPORAL_OPTION_DRY_RUN ")), None)
    report = json.loads(line.split(" ", 1)[1]) if line else None
    return {
        "command": command,
        "returncode": completed.returncode,
        "stderr": completed.stderr[-1000:],
        "report": report,
    }


def main() -> int:
    checks: list[dict[str, Any]] = []
    before_isaacgym = any(name == "isaacgym" or name.startswith("isaacgym.")
                          for name in sys.modules)
    try:
        contract = adapter.load_main_contract()
        checks.append(_check(
            "main_contract_load", "PASS",
            "Loaded main six_expert_support_contract without simulator dependencies.",
            contract_path=str(adapter._contract_path()),
            schema=contract.SCHEMA,
        ))
    except Exception as error:  # pragma: no cover - surfaced in JSON
        checks.append(_check("main_contract_load", "FAIL", str(error)))
        print(json.dumps({"status": "NOT_READY", "checks": checks}, indent=2))
        return 1

    rows = _synthetic_rows()
    try:
        manifest = adapter.build_manifest(rows)
        checks.append(_check(
            "synthetic_contract_fixture", "PASS",
            "Canonical synthetic rows satisfy all validator invariants; fixture is not persisted or used as data.",
            rows=len(rows), summary=manifest["summary"],
        ))
    except Exception as error:  # pragma: no cover - surfaced in JSON
        checks.append(_check("synthetic_contract_fixture", "FAIL", str(error)))

    source = EVALUATOR.read_text()
    router_source = ROUTER_EVALUATOR.read_text()
    source_syntax_ok = True
    syntax_error = None
    try:
        import ast
        ast.parse(source, filename=str(EVALUATOR))
        ast.parse(router_source, filename=str(ROUTER_EVALUATOR))
    except SyntaxError as error:
        source_syntax_ok = False
        syntax_error = str(error)
    checks.append(_check(
        "source_syntax", "PASS" if source_syntax_ok else "FAIL",
        "Temporal and router evaluator sources parse without importing Isaac Gym."
        if source_syntax_ok else "Evaluator source syntax failed.",
        error=syntax_error,
    ))
    audit = adapter.audit_legacy_evaluator(source)
    checks.append(_check(
        "legacy_evaluator_field_wiring", "PASS" if audit["ready"] else "FAIL",
        "Legacy evaluator emits every canonical field." if audit["ready"] else
        "Legacy evaluator lacks canonical observation/pose/router/provenance fields.",
        missing_source_markers=audit["missing_source_markers"],
        router_disabled_or_rejected=audit["router_disabled_or_rejected"],
    ))

    legacy_row = {
        "env_id": 0,
        "state": [0.0] * 49,
        "history_state": [[0.0] * 49] * 10,
        "history_action": [[0.0] * 18] * 10,
        "history_contact": [False] * 10,
        "candidate_actions": [[0.0] * 18] * 6,
        "assignment": 0,
        "assignment_propensity": 1 / 6,
        "option_executed_action": [[0.0] * 18] * 10,
        "future_contact_mask": [False] * 20,
        "start_frame": 0,
        "motion_name": "s3_airplane_lift",
    }
    try:
        adapter.adapt_record(legacy_row)
    except Exception as error:
        fail_closed = "object_pose_t_object_local_frame" in str(error)
        checks.append(_check(
            "legacy_adapter_fail_closed", "PASS" if fail_closed else "FAIL",
            "Legacy row is rejected with explicit missing canonical fields; no values are fabricated."
            if fail_closed else "Legacy adapter rejected the row without naming the missing pose contract.",
            error=str(error),
        ))
    else:
        checks.append(_check(
            "legacy_adapter_fail_closed", "FAIL",
            "Legacy row unexpectedly passed the canonical adapter."))

    config = json.loads(COLLECTOR_CONFIG.read_text())
    route_config = json.loads(ROUTE_CONFIG.read_text())
    expected = {
        "fit": {"simulator_seed": 256, "assignment_seed": 20260928256,
                "num_envs": 192, "minimum_valid_rows_per_arm": 30},
        "holdout": {"simulator_seed": 257, "assignment_seed": 20260928257,
                     "num_envs": 192, "minimum_valid_rows_per_arm": 30},
    }
    split_drift = {
        split: {
            key: {"expected": value, "actual": config.get("collections", {})
                  .get(split, {}).get(key)}
            for key, value in spec.items()
            if config.get("collections", {}).get(split, {}).get(key) != value
        }
        for split, spec in expected.items()
    }
    split_drift = {split: values for split, values in split_drift.items() if values}
    checks.append(_check(
        "new_split_contract", "PASS" if not split_drift else "FAIL",
        "Collector config matches dispatched 256/257 split contract." if not split_drift else
        "Collector config still owns legacy 254/255 semantics; old contract cannot be reused.",
        differences=split_drift,
    ))

    missing_artifacts = []
    artifact_hashes = {}
    for name, spec in route_config.get("experts", {}).items():
        path = _artifact_path(spec["checkpoint"])
        if not path.is_file():
            missing_artifacts.append({"kind": "expert_checkpoint", "name": name,
                                      "path": spec["checkpoint"]})
        else:
            actual = _sha256(path)
            artifact_hashes[f"expert:{name}"] = actual
            if actual != spec["sha256"]:
                missing_artifacts.append({"kind": "expert_checkpoint_hash_drift",
                                          "name": name, "expected": spec["sha256"],
                                          "actual": actual})
    for motion in route_config.get("motions", []):
        path = _artifact_path(motion["path"]) / "interaction_hand_inspire.pt"
        if not path.is_file():
            missing_artifacts.append({"kind": "motion_tensor", "name": motion["name"],
                                      "path": str(path)})
        else:
            actual = _sha256(path)
            artifact_hashes[f"motion:{motion['name']}"] = actual
            if actual != motion["interaction_hand_sha256"]:
                missing_artifacts.append({"kind": "motion_hash_drift", "name": motion["name"],
                                          "expected": motion["interaction_hand_sha256"],
                                          "actual": actual})
    router_expected = "1fa84c94891d63824be0100b5eb78f4aa1e37825c6d71fff65e517c0c7f2fc14"
    if ROUTER_MODEL.is_file():
        router_actual = _sha256(ROUTER_MODEL)
        artifact_hashes["frozen_c1_router"] = router_actual
        if router_actual != router_expected:
            missing_artifacts.append({"kind": "router_hash_drift", "expected": router_expected,
                                      "actual": router_actual})
    else:
        missing_artifacts.append({"kind": "frozen_c1_router", "path": str(ROUTER_MODEL),
                                  "expected_sha256": router_expected})
    checks.append(_check(
        "artifact_and_router_visibility", "PASS" if not missing_artifacts else "FAIL",
        "All frozen checkpoints, motion tensors and C1 router are visible and hash-matched."
        if not missing_artifacts else
        "Frozen checkpoint/motion/router provenance is unavailable; evaluator repair cannot be verified.",
        missing=missing_artifacts, visible_hashes=artifact_hashes,
    ))
    router_api_ok = all(marker in router_source for marker in (
        "--observation-router-model", "n_features_in_ != 1442",
        "self.observation_router.predict"))
    temporal_disables_router = "routed.MODEL_PATH = None" in source
    checks.append(_check(
        "c1_router_source_gate", "PASS" if router_api_ok and not temporal_disables_router else "FAIL",
        "Temporal evaluator retains the frozen C1 router path." if router_api_ok and not temporal_disables_router else
        "Router evaluator supports C1, but temporal evaluator disables it or the API markers are incomplete.",
        router_api_supported=router_api_ok,
        temporal_router_disabled=temporal_disables_router,
    ))
    admission_markers = (
        "validate_support_payload(",
        '"post_option_policy": collector["temporal_contract"]',
        '"route_expert": [collector["base_expert"]]',
        '"router_teacher_source": ["c1_observation_router"]',
        "torch.save(payload, runtime[\"record_output\"])",
    )
    marker_presence = {marker: marker in source for marker in admission_markers}
    admission_order_ok = (
        source.find("validate_support_payload(") >= 0 and
        source.find("validate_support_payload(") <
        source.find("torch.save(payload, runtime[\"record_output\"])")
    )
    live_admission_ok = all(marker_presence.values()) and admission_order_ok
    checks.append(_check(
        "live_main_contract_admission", "PASS" if live_admission_ok else "FAIL",
        "Canonical payload is sent through the strict support adapter before torch.save."
        if live_admission_ok else
        "Live evaluator save path is missing a strict pre-save support admission gate.",
        markers=marker_presence, validation_before_save=admission_order_ok,
    ))

    dry_run = _dry_run()
    report = dry_run["report"] or {}
    dry_run_ok = dry_run["returncode"] == 0 and report.get("isaacgym_imported") is False
    checks.append(_check(
        "legacy_evaluator_cpu_dry_run", "PASS" if dry_run_ok else "FAIL",
        "Import-safe evaluator dry-run completed without Isaac Gym." if dry_run_ok else
        "Evaluator CPU dry-run failed or imported Isaac Gym.",
        command=dry_run["command"], returncode=dry_run["returncode"],
        report=report, stderr=dry_run["stderr"],
    ))

    checks.append(_check(
        "isaacgym_import_guard", "PASS" if not before_isaacgym else "FAIL",
        "No Isaac Gym module was present during CPU preflight." if not before_isaacgym else
        "Isaac Gym was already imported before preflight.",
    ))
    failed = [item["name"] for item in checks if item["status"] == "FAIL"]
    ready = not failed
    label = "READY_FOR_COLLECTION" if ready else "NOT_READY"
    result = {
        "schema": "ref2dex.six_expert_support_wiring_preflight.v1",
        "status": label,
        "label": label,
        "collector_started": False,
        "gpu_started": False,
        "isaacgym_imported": False,
        "training_started": False,
        "synthetic_fixture_only": True,
        "hashes": {
            "evaluator_sha256": _sha256(EVALUATOR),
            "router_evaluator_sha256": _sha256(ROUTER_EVALUATOR),
            "collector_config_sha256": _sha256(COLLECTOR_CONFIG),
            "route_config_sha256": _sha256(ROUTE_CONFIG),
            "main_contract_sha256": _sha256(adapter._contract_path()),
            "visible_artifacts": artifact_hashes,
        },
        "resource_evidence": {
            "gpu_process_started": False,
            "isaacgym_imported": False,
            "collector_process_started": False,
            "training_process_started": False,
            "dry_run_child_returned": dry_run["returncode"],
        },
        "checks": checks,
        "failed_checks": failed,
        "next": "collector remains blocked until canonical preflight is repaired" if failed else
                 "root may separately review a collector dispatch; this command performed no collection",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
