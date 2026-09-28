"""CPU-only contract for provenance-complete six-expert support collection.

This module validates records before a collector or fitter is allowed to use
them.  It deliberately has no torch, simulator, filesystem, or GPU dependency.
The contract is fail-closed: a record must carry the full pre-action input,
six candidate actions and checkpoint provenance, the actually executed arm,
object-local one-step target, five contact flags, and the frozen C1 router
teacher provenance.  ``build_manifest`` only returns a machine-readable
summary; it never writes a manifest or creates data.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


SCHEMA = "ref2dex.six_expert_support_contract.v1"
EXPERT_COUNT = 6
ACTION_DIM = 18
OBSERVATION_DIM = 1442
CONTACT_HORIZON = 5
MIN_ROWS_PER_ARM = 30
PROPENSITY = 1.0 / EXPERT_COUNT
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OBJECT_LIFT_AXIS_METADATA = {
    "world_axis": [0.0, 0.0, 1.0],
    "frame": "object_local_at_trigger_t",
    "source": "inverse_rotation_world_z_using_trigger_object_quaternion_xyzw",
    "conversion": "local_translation_target_inverse_quaternion_xyzw",
}

REQUIRED_FIELDS = frozenset(
    {
        "episode_id",
        "split",
        "pre_action_observation",
        "object_pose_t_object_local_frame",
        "candidate_actions",
        "candidate_expert_names_and_checkpoint_sha256",
        "assignment",
        "assignment_propensity",
        "executed_action",
        "object_pose_t_plus_1_object_local_frame",
        "target_delta_object_local_1",
        "object_lift_axis",
        "contact_mask_t_plus_1_to_t_plus_5",
        "router_teacher_candidate_id",
        "router_teacher_action",
        "router_model_sha256",
        "router_input_state_sha256",
        "router_teacher_source",
        "start_frame",
        "motion_name",
        "route_config_sha256",
        "collector_config_sha256",
    }
)


class ContractError(ValueError):
    """Raised when a support record or collection manifest is unsafe to use."""


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _finite_vector(value: Any, *, name: str, length: int | None = None) -> tuple[float, ...]:
    _fail(
        isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)),
        f"{name} must be a list or tuple",
    )
    if length is not None:
        _fail(len(value) == length, f"{name} must have length {length}")
    result: list[float] = []
    for item in value:
        _fail(
            isinstance(item, (int, float)) and not isinstance(item, bool),
            f"{name} must contain numeric values",
        )
        converted = float(item)
        _fail(math.isfinite(converted), f"{name} contains a non-finite value")
        result.append(converted)
    _fail(bool(result), f"{name} must not be empty")
    return tuple(result)


def _sha256(value: Any, *, name: str) -> str:
    _fail(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
          f"{name} must be a lowercase SHA256 hex digest")
    return value


def _integer(value: Any, *, name: str, minimum: int | None = None,
             maximum: int | None = None) -> int:
    _fail(isinstance(value, int) and not isinstance(value, bool), f"{name} must be an integer")
    if minimum is not None:
        _fail(value >= minimum, f"{name} must be >= {minimum}")
    if maximum is not None:
        _fail(value <= maximum, f"{name} must be <= {maximum}")
    return value


def _same_vector(left: Sequence[float], right: Sequence[float], *, name: str) -> None:
    _fail(len(left) == len(right), f"{name} shape drift")
    _fail(
        all(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
            for a, b in zip(left, right)),
        f"{name} does not match its candidate action",
    )


def _validate_experts(value: Any) -> tuple[tuple[str, str], ...]:
    _fail(isinstance(value, (list, tuple)) and len(value) == EXPERT_COUNT,
          "candidate expert provenance must contain six entries")
    result: list[tuple[str, str]] = []
    names: set[str] = set()
    for index, entry in enumerate(value):
        _fail(isinstance(entry, Mapping), f"candidate expert {index} must be an object")
        name = entry.get("name")
        _fail(isinstance(name, str) and bool(name), f"candidate expert {index} name is invalid")
        _fail(name not in names, f"candidate expert names must be unique: {name}")
        names.add(name)
        digest = _sha256(entry.get("checkpoint_sha256"),
                         name=f"candidate expert {index} checkpoint_sha256")
        result.append((name, digest))
    return tuple(result)


def _validate_pose(value: Any, *, name: str) -> tuple[float, ...]:
    # A position-only object-local pose (xyz) or xyz+quaternion pose is valid;
    # no other shape can be interpreted without an unstated transform contract.
    pose = _finite_vector(value, name=name)
    _fail(len(pose) in {3, 7}, f"{name} must contain xyz or xyz+quaternion (3 or 7 values)")
    return pose


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one collection row and return its normalized provenance summary."""
    _fail(isinstance(record, Mapping), "record must be an object")
    missing = sorted(REQUIRED_FIELDS.difference(record))
    _fail(not missing, f"record is missing required fields: {missing}")

    episode_id = record["episode_id"]
    _fail(isinstance(episode_id, str) and bool(episode_id), "episode_id must be non-empty")
    split = record["split"]
    _fail(split in {"fit", "holdout"}, "split must be fit or holdout")

    observation = _finite_vector(record["pre_action_observation"],
                                 name="pre_action_observation", length=OBSERVATION_DIM)
    poses_t = _validate_pose(record["object_pose_t_object_local_frame"],
                             name="object_pose_t_object_local_frame")
    poses_next = _validate_pose(record["object_pose_t_plus_1_object_local_frame"],
                                name="object_pose_t_plus_1_object_local_frame")
    _fail(len(poses_t) == len(poses_next), "object pose shape drift between t and t+1")
    delta = _finite_vector(record["target_delta_object_local_1"],
                           name="target_delta_object_local_1", length=3)
    expected_delta = tuple(poses_next[index] - poses_t[index] for index in range(3))
    _fail(
        all(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
            for actual, expected in zip(delta, expected_delta)),
        "target_delta_object_local_1 does not match object-local pose difference",
    )
    axis = _finite_vector(record["object_lift_axis"],
                          name="object_lift_axis", length=3)
    _fail(
        math.isclose(sum(value * value for value in axis), 1.0,
                     rel_tol=0.0, abs_tol=1e-5),
        "object_lift_axis must be unit length",
    )

    candidate_actions_value = record["candidate_actions"]
    _fail(isinstance(candidate_actions_value, (list, tuple)) and
          len(candidate_actions_value) == EXPERT_COUNT,
          "candidate_actions must have shape [6, 18]")
    candidate_actions = tuple(
        _finite_vector(action, name=f"candidate_actions[{index}]", length=ACTION_DIM)
        for index, action in enumerate(candidate_actions_value)
    )
    experts = _validate_experts(record["candidate_expert_names_and_checkpoint_sha256"])

    assignment = _integer(record["assignment"], name="assignment",
                          minimum=0, maximum=EXPERT_COUNT - 1)
    propensity = record["assignment_propensity"]
    _fail(isinstance(propensity, (int, float)) and not isinstance(propensity, bool),
          "assignment_propensity must be numeric")
    _fail(math.isclose(float(propensity), PROPENSITY, rel_tol=0.0, abs_tol=1e-12),
          "assignment_propensity must equal 1/6")
    executed = _finite_vector(record["executed_action"], name="executed_action",
                              length=ACTION_DIM)
    _same_vector(executed, candidate_actions[assignment], name="executed_action")

    contact = record["contact_mask_t_plus_1_to_t_plus_5"]
    _fail(isinstance(contact, (list, tuple)) and len(contact) == CONTACT_HORIZON,
          "contact_mask_t_plus_1_to_t_plus_5 must have length 5")
    for index, flag in enumerate(contact):
        _fail(isinstance(flag, (bool, int)) and not isinstance(flag, float) and
              int(flag) in {0, 1},
              f"contact flag {index} must be boolean")

    teacher_id = _integer(record["router_teacher_candidate_id"],
                          name="router_teacher_candidate_id",
                          minimum=0, maximum=EXPERT_COUNT - 1)
    teacher_action = _finite_vector(record["router_teacher_action"],
                                    name="router_teacher_action", length=ACTION_DIM)
    _same_vector(teacher_action, candidate_actions[teacher_id], name="router_teacher_action")
    _fail(record["router_teacher_source"] == "c1_observation_router",
          "router_teacher_source must be c1_observation_router; fallback is not allowed")
    router_model = _sha256(record["router_model_sha256"], name="router_model_sha256")
    router_input = _sha256(record["router_input_state_sha256"],
                           name="router_input_state_sha256")

    _integer(record["start_frame"], name="start_frame", minimum=0)
    motion_name = record["motion_name"]
    _fail(isinstance(motion_name, str) and bool(motion_name), "motion_name must be non-empty")
    route_hash = _sha256(record["route_config_sha256"], name="route_config_sha256")
    collector_hash = _sha256(record["collector_config_sha256"], name="collector_config_sha256")

    return {
        "episode_id": episode_id,
        "split": split,
        "assignment": assignment,
        "candidate_experts": experts,
        "router_model_sha256": router_model,
        "route_config_sha256": route_hash,
        "collector_config_sha256": collector_hash,
        "observation_dim": len(observation),
        "object_lift_axis_metadata": OBJECT_LIFT_AXIS_METADATA,
    }


def validate_records(records: Iterable[Mapping[str, Any]], *,
                     min_rows_per_arm: int = MIN_ROWS_PER_ARM,
                     expected_provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate all rows, provenance consistency, split disjointness and arm support."""
    _fail(isinstance(min_rows_per_arm, int) and min_rows_per_arm > 0,
          "min_rows_per_arm must be a positive integer")
    materialized = list(records)
    _fail(materialized, "at least one support record is required")
    summaries = [validate_record(record) for record in materialized]

    episode_ids = [summary["episode_id"] for summary in summaries]
    _fail(len(set(episode_ids)) == len(episode_ids),
          "episode_id values must be globally unique across fit and holdout")

    def consistent(key: str) -> Any:
        values = {summary[key] for summary in summaries}
        _fail(len(values) == 1, f"{key} drift across support records")
        return next(iter(values))

    experts = consistent("candidate_experts")
    router_model = consistent("router_model_sha256")
    route_hash = consistent("route_config_sha256")
    collector_hash = consistent("collector_config_sha256")

    expected = dict(expected_provenance or {})
    for key, actual in (
        ("candidate_experts", experts),
        ("router_model_sha256", router_model),
        ("route_config_sha256", route_hash),
        ("collector_config_sha256", collector_hash),
    ):
        if key in expected:
            _fail(expected[key] == actual, f"{key} does not match expected provenance")
    if "object_lift_axis" in expected:
        _fail(expected["object_lift_axis"] == OBJECT_LIFT_AXIS_METADATA,
              "object_lift_axis provenance differs from canonical metadata")

    arm_counts = {
        split: {str(arm): 0 for arm in range(EXPERT_COUNT)}
        for split in ("fit", "holdout")
    }
    split_episode_ids: dict[str, set[str]] = {"fit": set(), "holdout": set()}
    for summary in summaries:
        split = summary["split"]
        arm = str(summary["assignment"])
        arm_counts[split][arm] += 1
        split_episode_ids[split].add(summary["episode_id"])
    for split in ("fit", "holdout"):
        _fail(split_episode_ids[split], f"{split} split must not be empty")
        underflow = {arm: count for arm, count in arm_counts[split].items()
                     if count < min_rows_per_arm}
        _fail(not underflow,
              f"{split} arm support below {min_rows_per_arm}: {underflow}")
    _fail(split_episode_ids["fit"].isdisjoint(split_episode_ids["holdout"]),
          "fit and holdout episode IDs must be disjoint")

    return {
        "schema": SCHEMA,
        "rows": len(summaries),
        "observation_dim": OBSERVATION_DIM,
        "candidate_action_shape": [EXPERT_COUNT, ACTION_DIM],
        "contact_horizon": CONTACT_HORIZON,
        "propensity": PROPENSITY,
        "min_rows_per_arm": min_rows_per_arm,
        "arm_counts": arm_counts,
        "fit_episodes": len(split_episode_ids["fit"]),
        "holdout_episodes": len(split_episode_ids["holdout"]),
        "candidate_experts": [
            {"name": name, "checkpoint_sha256": digest}
            for name, digest in experts
        ],
        "router_model_sha256": router_model,
        "route_config_sha256": route_hash,
        "collector_config_sha256": collector_hash,
        "object_lift_axis_metadata": OBJECT_LIFT_AXIS_METADATA,
    }


def build_manifest(records: Iterable[Mapping[str, Any]], *,
                   expected_provenance: Mapping[str, Any] | None = None,
                   min_rows_per_arm: int = MIN_ROWS_PER_ARM) -> dict[str, Any]:
    """Return a collector preflight manifest after validating every record."""
    summary = validate_records(records, min_rows_per_arm=min_rows_per_arm,
                               expected_provenance=expected_provenance)
    return {
        "schema": SCHEMA,
        "status": "READY_FOR_COLLECTOR_PREFLIGHT",
        "contract": {
            "pre_action_observation_dim": OBSERVATION_DIM,
            "candidate_actions_shape": [EXPERT_COUNT, ACTION_DIM],
            "assignment_propensity": "1/6",
            "contact_horizon": CONTACT_HORIZON,
            "router_teacher_source": "c1_observation_router",
            "fit_holdout_episode_disjoint": True,
            "minimum_valid_rows_per_arm": min_rows_per_arm,
            "object_lift_axis": OBJECT_LIFT_AXIS_METADATA,
        },
        "summary": summary,
    }


__all__ = [
    "ACTION_DIM",
    "CONTACT_HORIZON",
    "ContractError",
    "EXPERT_COUNT",
    "MIN_ROWS_PER_ARM",
    "OBSERVATION_DIM",
    "OBJECT_LIFT_AXIS_METADATA",
    "PROPENSITY",
    "SCHEMA",
    "build_manifest",
    "validate_record",
    "validate_records",
]
