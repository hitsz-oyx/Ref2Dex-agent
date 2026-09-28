"""CPU-only contract tests for the HF02 temporal-option evaluator."""
from __future__ import annotations

import json
import ast
from pathlib import Path
import os
import subprocess
import sys

import pytest
import torch

from src.task.CmResidual.temporal_option_contract import (
    ContractError,
    balanced_environment_assignment,
    eligible_trigger_mask,
    git_blob_sha1,
    option_active_mask,
    record_future_step,
    sha256_path,
    validate_frozen_contract,
    validate_record_payload,
)


ROOT = Path(__file__).resolve().parents[4]
COLLECTOR = ROOT / "src/task/CmResidual/configs/airplane_temporal_expert_probe.json"
ROUTE = ROOT / "src/task/CmResidual/configs/hf02_temporal_canonical_route.json"
EVALUATOR = ROOT / "third_party/DExplore/dexplore/evaluate_temporal_expert_option.py"


def _provenance():
    return validate_frozen_contract(ROOT, COLLECTOR, ROUTE, verify_artifacts=False)


def test_frozen_route_and_collector_are_exact_six_arm_contract():
    provenance = _provenance()
    collector = provenance["collector"]
    route = provenance["route"]
    assert route["route_mode"] == "simulator_object_id"
    assert route["objects"] == ["airplane"]
    assert collector["candidate_experts"] == [
        "balanced_e360", "cup_e340", "duck_e340",
        "mixed12_e300", "source_e260", "train5_e320"]
    assert collector["motions"] == [
        "s3_airplane_lift", "s7_airplane_lift_Retake", "s9_airplane_lift"]
    assert collector["assignment"]["propensity_numerator"] == 1
    assert collector["assignment"]["propensity_denominator"] == 6
    assert collector["temporal_contract"] == {
        "history_steps": 10,
        "option_steps": 10,
        "future_steps": 20,
        "trigger": "first_valid_hand_object_contact_after_complete_history",
        "first_episode_only": True,
        "post_option_policy": "canonical_route_expert",
    }


def test_seeded_assignment_is_reproducible_balanced_and_six_arms():
    first = balanced_environment_assignment(192, 6, 20260928256)
    second = balanced_environment_assignment(192, 6, 20260928256)
    other = balanced_environment_assignment(192, 6, 20260928257)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert sorted(first.unique().tolist()) == list(range(6))
    assert [int((first == arm).sum()) for arm in range(6)] == [32] * 6


def test_trigger_requires_ten_history_and_option_mask_has_ten_steps():
    contact = torch.tensor([True])
    triggered = torch.tensor([False])
    ended = torch.tensor([False])
    reset = torch.tensor([False])
    progress = torch.tensor([11])
    assert not bool(eligible_trigger_mask(
        contact, triggered, ended, reset, progress, torch.tensor([9]))[0])
    eligible = eligible_trigger_mask(
        contact, triggered, ended, reset, progress, torch.tensor([10]))
    assert bool(eligible[0])
    triggered[:] = True
    elapsed = torch.tensor([0])
    active = []
    for _ in range(11):
        active.append(bool(option_active_mask(triggered, ended, elapsed, 10)[0]))
        elapsed += 1
    assert active == [True] * 10 + [False]


def test_future_boundary_rejects_early_reset_but_accepts_terminal_twentieth_step():
    def state():
        return {
            "triggered": torch.tensor([True]),
            "episode_ended": torch.tensor([False]),
            "future_valid": torch.tensor([True]),
            "future_elapsed": torch.tensor([0]),
            "future_contact_mask": torch.zeros(1, 20, dtype=torch.bool),
            "future_contact_supported_lift_m": torch.zeros(1, 20),
        }

    early = state()
    record_future_step(**early, post_contact=torch.tensor([True]),
                       lift_m=torch.tensor([0.01]), done=torch.tensor([True]))
    assert int(early["future_elapsed"][0]) == 1
    assert not bool(early["future_valid"][0])
    assert bool(early["future_contact_mask"][0, 0])
    early["episode_ended"][:] = True
    active = record_future_step(
        **early, post_contact=torch.tensor([True]), lift_m=torch.tensor([0.50]),
        done=torch.tensor([False]))
    assert not bool(active[0])
    assert int(early["future_elapsed"][0]) == 1
    assert float(early["future_contact_supported_lift_m"][0, 1]) == 0.0

    complete = state()
    for step in range(20):
        done = torch.tensor([step == 19])
        record_future_step(
            **complete, post_contact=torch.tensor([step % 2 == 0]),
            lift_m=torch.tensor([0.03 + step / 1000.0]), done=done)
    assert int(complete["future_elapsed"][0]) == 20
    assert bool(complete["future_valid"][0])
    assert bool(complete["future_contact_mask"][0, 19]) is False
    assert bool(complete["future_contact_mask"][0, 18]) is True


def _synthetic_payload():
    provenance = _provenance()
    collector = provenance["collector"]
    n = 6
    experts = collector["candidate_experts"]
    assignment = torch.arange(6, dtype=torch.int8)
    candidate = torch.arange(n * 6 * 18, dtype=torch.float32).reshape(n, 6, 18)
    option = candidate[torch.arange(n), assignment.long()].unsqueeze(1).repeat(1, 10, 1)
    contact = torch.zeros(n, 20, dtype=torch.bool)
    contact[:, ::2] = True
    supported = torch.where(contact, torch.full((n, 20), 0.04), torch.zeros(n, 20))
    teacher_id = (assignment.long() + 1) % 6
    records = {
        "env_id": torch.arange(n),
        "motion_id": torch.tensor([0, 1, 2, 0, 1, 2]),
        "motion_name": [collector["motions"][i % 3] for i in range(n)],
        "object_name": ["airplane"] * n,
        "simulator_object_id": torch.zeros(n, dtype=torch.long),
        "route_expert": ["source_e260"] * n,
        "assignment": assignment,
        "assignment_propensity": torch.full((n,), 1 / 6),
        "trigger_step": torch.full((n,), 10),
        "start_frame": torch.arange(n),
        "state": torch.zeros(n, 49),
        "base_action": torch.zeros(n, 18),
        "candidate_actions": candidate,
        "history_state": torch.zeros(n, 10, 49),
        "history_action": torch.zeros(n, 10, 18),
        "history_contact": torch.zeros(n, 10, dtype=torch.bool),
        "option_candidate_action": option,
        "option_executed_action": option.clone(),
        "future_contact_mask": contact,
        "episode_id": [f"fit-test-episode-{i}" for i in range(n)],
        "split": ["fit"] * n,
        "pre_action_observation": torch.zeros(n, 1442),
        "object_pose_t_object_local_frame": torch.zeros(n, 3),
        "object_pose_t_plus_1_object_local_frame": torch.tensor(
            [[0.01, -0.02, 0.03]] * n),
        "target_delta_object_local_1": torch.tensor(
            [[0.01, -0.02, 0.03]] * n),
        "contact_mask_t_plus_1_to_t_plus_5": contact[:, :5].clone(),
        "candidate_expert_names_and_checkpoint_sha256": [
            [{"name": name, "checkpoint_sha256": provenance["expert_checkpoint_sha256"][name]}
             for name in experts] for _ in range(n)],
        "executed_action": candidate[torch.arange(n), assignment.long()].clone(),
        "router_teacher_candidate_id": teacher_id.to(torch.int8),
        "router_teacher_action": candidate[torch.arange(n), teacher_id].clone(),
        "router_model_sha256": ["3" * 64] * n,
        "router_input_state_sha256": ["4" * 64] * n,
        "router_teacher_source": ["c1_observation_router"] * n,
        "route_config_sha256": [provenance["route_config_sha256"]] * n,
        "collector_config_sha256": [provenance["collector_config_sha256"]] * n,
        "future_contact_supported_lift_m": supported,
        "followup_contact_fraction": contact.float().mean(1),
        "followup_max_contact_lift_m": supported.max(1).values,
        "final_lift_success": torch.ones(n, dtype=torch.bool),
        "final_max_contact_lift_m": torch.full((n,), 0.04),
        "final_contact_fraction": torch.full((n,), 0.5),
        "final_episode_steps": torch.full((n,), 40, dtype=torch.long),
    }
    return {
        "schema": "ref2dex.temporal_expert_option.v2",
        "run_status": "COMPLETED",
        "candidate_experts": experts,
        "base_expert": "source_e260",
        "post_option_policy": "canonical_route_expert",
        "history_steps": 10,
        "option_steps": 10,
        "future_steps": 20,
        "provenance": {
            "collector_config_sha256": provenance["collector_config_sha256"],
            "route_config_sha256": provenance["route_config_sha256"],
            "canonical_route_sha256": provenance["route_config_sha256"],
            "expert_checkpoint_sha256": provenance["expert_checkpoint_sha256"],
            "motion_sha256": provenance["motion_sha256"],
            "evaluator_sha256": "a" * 64,
            "evaluator_git_blob_sha1": "b" * 40,
        },
        "records": records,
    }, collector, provenance


def test_record_schema_accepts_valid_payload_and_rejects_propensity_drift():
    payload, collector, provenance = _synthetic_payload()
    summary = validate_record_payload(payload, collector, provenance)
    assert summary["rows"] == 6
    assert summary["arm_counts"] == {
        "balanced_e360": 1, "cup_e340": 1, "duck_e340": 1,
        "mixed12_e300": 1, "source_e260": 1, "train5_e320": 1,
    }
    payload["records"]["assignment_propensity"][0] = 0.5
    with pytest.raises(ContractError, match="propensity"):
        validate_record_payload(payload, collector, provenance)


def test_record_schema_rejects_silent_source_continuation():
    payload, collector, provenance = _synthetic_payload()
    payload["records"]["option_executed_action"][0, 4, 0] += 1.0
    with pytest.raises(ContractError, match="assigned option"):
        validate_record_payload(payload, collector, provenance)


def test_hash_helpers_match_known_bytes(tmp_path):
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"hf02-contract")
    assert sha256_path(path) == (
        "d8e579d16959503e1cac2c2226a47ced2b493411a1231fa7868008a0e6d41850")
    assert git_blob_sha1(path) == "b25663806376c9f1e6827bb5c2604374239163bf"


def test_canonical_route_byte_drift_is_rejected(tmp_path):
    changed = tmp_path / "route.json"
    changed.write_text(ROUTE.read_text() + "\n")
    with pytest.raises(ContractError, match="route SHA256 drift"):
        validate_frozen_contract(ROOT, COLLECTOR, changed,
                                 verify_artifacts=False)


def test_cpu_dry_run_does_not_import_isaacgym():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, str(EVALUATOR), "--dry-run", "--skip-artifact-hashes"],
        cwd=ROOT, env=env, check=True, text=True, capture_output=True)
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("REF2DEX_TEMPORAL_OPTION_DRY_RUN "))
    report = json.loads(line.split(" ", 1)[1])
    assert report["isaacgym_imported"] is False
    assert report["route_config_sha256"].startswith("afedfa54")
    assert report["assignments"]["fit"]["arm_counts"] == {
        str(index): 32 for index in range(6)}


def _temporal_get_action_source():
    tree = ast.parse(EVALUATOR.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TemporalOptionPlayer":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "get_action":
                    return child, ast.get_source_segment(EVALUATOR.read_text(), child)
    raise AssertionError("TemporalOptionPlayer.get_action not found")


def test_temporal_get_action_records_initial_c1_route_before_post_option_route():
    method, source = _temporal_get_action_source()
    calls = [
        node.func.attr for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "_record_initial_observation_route" in calls
    assert "_candidate_actions" in calls
    assert source.index("_record_initial_observation_route") < source.index(
        "route_by_motion")
    assert "_router_teacher(" in source
    assert source.index("_router_teacher(") > source.index("route_by_motion")
    assert "route_by_motion[task.data_id.long()]" in source


def test_temporal_source_keeps_trigger_teacher_distinct_from_initial_route():
    source = EVALUATOR.read_text()
    assert "def _record_initial_observation_route" in source
    assert "self.initial_expert_names" in source
    assert "self.initial_route_choice = choice.detach().clone()" in source
    assert "self.trigger_router_teacher_candidate_id" in source
    assert "self._router_teacher(" in source
    assert '"router_teacher_source": ["c1_observation_router"]' in source
    assert '"route_expert": [collector["base_expert"]]' in source
