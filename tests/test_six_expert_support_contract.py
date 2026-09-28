from __future__ import annotations

import copy

import pytest

from src.task.CmResidual.six_expert_support_contract import (
    ContractError,
    build_manifest,
    validate_records,
)


HASHES = {
    "route_config_sha256": "1" * 64,
    "collector_config_sha256": "2" * 64,
    "router_model_sha256": "3" * 64,
}
EXPERTS = [
    {"name": f"expert_{index}", "checkpoint_sha256": str(index + 4) * 64}
    for index in range(6)
]


def _record(split: str, arm: int, ordinal: int) -> dict:
    pose_t = [float(ordinal), 0.1, -0.2]
    pose_next = [pose_t[0] + 0.01, pose_t[1] - 0.02, pose_t[2] + 0.03]
    candidates = [[float(candidate + component / 1000.0) for component in range(18)]
                  for candidate in range(6)]
    return {
        "episode_id": f"{split}-seed-{split}-episode-{ordinal}",
        "split": split,
        "pre_action_observation": [0.0] * 1442,
        "object_pose_t_object_local_frame": pose_t,
        "candidate_actions": candidates,
        "candidate_expert_names_and_checkpoint_sha256": copy.deepcopy(EXPERTS),
        "assignment": arm,
        "assignment_propensity": 1 / 6,
        "executed_action": list(candidates[arm]),
        "object_pose_t_plus_1_object_local_frame": pose_next,
        "target_delta_object_local_1": [0.01, -0.02, 0.03],
        "contact_mask_t_plus_1_to_t_plus_5": [True, False, True, True, False],
        "router_teacher_candidate_id": (arm + 1) % 6,
        "router_teacher_action": list(candidates[(arm + 1) % 6]),
        "router_model_sha256": HASHES["router_model_sha256"],
        "router_input_state_sha256": "4" * 64,
        "router_teacher_source": "c1_observation_router",
        "start_frame": ordinal,
        "motion_name": "airplane_lift",
        "route_config_sha256": HASHES["route_config_sha256"],
        "collector_config_sha256": HASHES["collector_config_sha256"],
    }


def _valid_rows() -> list[dict]:
    return [
        _record(split, arm, ordinal=split_index * 180 + ordinal * 6 + arm)
        for split_index, split in enumerate(("fit", "holdout"))
        for ordinal in range(30)
        for arm in range(6)
    ]


def test_valid_synthetic_support_builds_machine_manifest():
    rows = _valid_rows()
    manifest = build_manifest(rows)
    assert manifest["status"] == "READY_FOR_COLLECTOR_PREFLIGHT"
    assert manifest["contract"]["pre_action_observation_dim"] == 1442
    assert manifest["contract"]["candidate_actions_shape"] == [6, 18]
    assert manifest["summary"]["arm_counts"] == {
        "fit": {str(arm): 30 for arm in range(6)},
        "holdout": {str(arm): 30 for arm in range(6)},
    }


def test_missing_field_is_rejected():
    row = _valid_rows()[0]
    del row["pre_action_observation"]
    with pytest.raises(ContractError, match="missing"):
        validate_records([row])


def test_hash_drift_is_rejected_before_arm_summary():
    rows = _valid_rows()
    rows[1]["route_config_sha256"] = "9" * 64
    with pytest.raises(ContractError, match="route_config_sha256 drift"):
        validate_records(rows)


def test_fit_holdout_episode_overlap_is_rejected():
    rows = _valid_rows()
    rows[180]["episode_id"] = rows[0]["episode_id"]
    with pytest.raises(ContractError, match="globally unique"):
        validate_records(rows)


def test_propensity_drift_is_rejected():
    rows = _valid_rows()
    rows[0]["assignment_propensity"] = 0.5
    with pytest.raises(ContractError, match="1/6"):
        validate_records(rows)


def test_non_router_fallback_is_rejected():
    rows = _valid_rows()
    rows[0]["router_teacher_source"] = "source_e260"
    with pytest.raises(ContractError, match="fallback"):
        validate_records(rows)


def test_arm_underflow_is_rejected():
    rows = [row for row in _valid_rows()
            if not (row["split"] == "fit" and row["assignment"] == 0 and
                    row["start_frame"] == 0)]
    with pytest.raises(ContractError, match="below 30"):
        validate_records(rows)
