"""CPU tests for live canonical support admission."""
from __future__ import annotations

import copy

import pytest
import torch

from src.task.CmResidual.cmlite import local_translation_target
from src.task.CmResidual.six_expert_support_adapter import validate_payload


def _payload(rows: int = 6):
    names = [
        "balanced_e360", "cup_e340", "duck_e340",
        "mixed12_e300", "source_e260", "train5_e320",
    ]
    checkpoint = {name: f"{index + 1:x}" * 64
                  for index, name in enumerate(names)}
    route_hash = "a" * 64
    collector_hash = "b" * 64
    router_hash = "c" * 64
    assignment = torch.arange(rows, dtype=torch.int64) % 6
    candidate = torch.arange(rows * 6 * 18, dtype=torch.float32).reshape(rows, 6, 18)
    teacher_id = (assignment + 1) % 6
    pose_t = torch.zeros(rows, 3)
    pose_next = torch.tensor([[0.0, -1.0, 0.0]] * rows)
    contact = torch.tensor([[True, False, True, False, True]] * rows)
    records = {
        "episode_id": [f"fit-live-{index}" for index in range(rows)],
        "split": ["fit"] * rows,
        "pre_action_observation": torch.zeros(rows, 1442),
        "object_pose_t_object_local_frame": pose_t,
        "object_pose_t_plus_1_object_local_frame": pose_next,
        "target_delta_object_local_1": pose_next - pose_t,
        "candidate_actions": candidate,
        "candidate_expert_names_and_checkpoint_sha256": [
            [{"name": name, "checkpoint_sha256": checkpoint[name]}
             for name in names] for _ in range(rows)],
        "assignment": assignment,
        "assignment_propensity": torch.full((rows,), 1 / 6),
        "executed_action": candidate[torch.arange(rows), assignment],
        "contact_mask_t_plus_1_to_t_plus_5": contact,
        "router_teacher_candidate_id": teacher_id,
        "router_teacher_action": candidate[torch.arange(rows), teacher_id],
        "router_model_sha256": [router_hash] * rows,
        "router_input_state_sha256": ["d" * 64] * rows,
        "router_teacher_source": ["c1_observation_router"] * rows,
        "start_frame": list(range(rows)),
        "motion_name": ["s3_airplane_lift"] * rows,
        "route_config_sha256": [route_hash] * rows,
        "collector_config_sha256": [collector_hash] * rows,
        # These fields are intentionally outside the main required set and
        # make the post-option route semantics explicit at the live boundary.
        "route_expert": ["source_e260"] * rows,
    }
    payload = {
        "schema": "ref2dex.temporal_expert_option.v2",
        "run_status": "COMPLETED",
        "candidate_experts": names,
        "base_expert": "source_e260",
        "post_option_policy": "canonical_route_expert",
        "records": records,
    }
    expected = {
        "expert_checkpoint_sha256": checkpoint,
        "router_model_sha256": router_hash,
        "route_config_sha256": route_hash,
        "collector_config_sha256": collector_hash,
    }
    return payload, expected


def test_live_admission_uses_main_rows_and_keeps_teacher_route_distinct():
    payload, expected = _payload()
    result = validate_payload(payload, expected_provenance=expected,
                              min_rows_per_arm=1)
    assert result["main_row_validation"] is True
    assert result["router_teacher_source"] == "c1_observation_router"
    assert result["post_option_route"] == "source_e260"
    assert result["arm_counts"] == {
        "balanced_e360": 1, "cup_e340": 1, "duck_e340": 1,
        "mixed12_e300": 1, "source_e260": 1, "train5_e320": 1,
    }

    fallback = copy.deepcopy(payload)
    fallback["records"]["router_teacher_source"][0] = "source_e260"
    with pytest.raises(ValueError, match="C1 teacher source"):
        validate_payload(fallback, expected_provenance=expected, min_rows_per_arm=1)

    route_drift = copy.deepcopy(payload)
    route_drift["records"]["route_expert"][0] = "cup_e340"
    with pytest.raises(ValueError, match="post-option route"):
        validate_payload(route_drift, expected_provenance=expected, min_rows_per_arm=1)


def test_live_admission_fails_closed_on_arm_underflow():
    payload, expected = _payload()
    with pytest.raises(ValueError, match="arm support below 2"):
        validate_payload(payload, expected_provenance=expected, min_rows_per_arm=2)


def test_object_local_delta_rotates_world_translation():
    # Object rotates +90 degrees around world z.  A world +x displacement is
    # object-local -y; accepting [1, 0, 0] would silently use world pose.
    half = 2 ** -0.5
    state_t = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, half, half, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0]])
    state_next = state_t.clone()
    state_next[:, 0] = 1.0
    local = local_translation_target(state_t, state_next)
    torch.testing.assert_close(local, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6, rtol=0)
    assert not torch.allclose(local, torch.tensor([[1.0, 0.0, 0.0]]))
