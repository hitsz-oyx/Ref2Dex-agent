"""Pure contract helpers for the HF02 temporal expert-option collector.

This module deliberately has no Isaac Gym dependency.  The evaluator and its
CPU tests share these checks so route, assignment, episode-boundary, and output
schema semantics cannot silently diverge.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch


COLLECTOR_SCHEMA = "ref2dex.temporal_expert_option_config.v2"
RECORD_SCHEMA = "ref2dex.temporal_expert_option.v2"
RUN_SCHEMA = "ref2dex.temporal_expert_option_run.v2"
ROUTE_SCHEMA = "ref2dex.hf02_temporal_canonical_route.v1"
STATE_DIM = 49
ACTION_DIM = 18


class ContractError(ValueError):
    """Raised before collection when the frozen HF02 contract differs."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def validate_frozen_contract(
        repo_root: Path,
        collector_path: Path,
        route_path: Path,
        *,
        verify_artifacts: bool = False,
        artifact_root: Path | None = None) -> dict:
    """Validate the exact v2 collector contract and return pinned provenance."""
    repo_root = repo_root.resolve()
    collector_path = collector_path.resolve()
    route_path = route_path.resolve()
    artifact_root = (artifact_root or repo_root).resolve()
    collector = _load_json(collector_path)
    route = _load_json(route_path)

    _require(collector.get("schema") == COLLECTOR_SCHEMA,
             "collector schema differs from v2")
    _require(collector.get("probe_id") == "P-20260926-temporal-expert-credit",
             "collector is not owned by the HF02 slot-2 card")
    frozen_route = collector.get("canonical_route", {})
    _require(frozen_route.get("path") ==
             "src/task/CmResidual/configs/hf02_temporal_canonical_route.json",
             "collector canonical route path differs")
    route_sha = sha256_path(route_path)
    _require(frozen_route.get("sha256") == route_sha,
             "canonical route SHA256 drift")
    _require(route.get("schema") == ROUTE_SCHEMA, "canonical route schema drift")
    _require(route.get("route_mode") == "simulator_object_id",
             "route must use simulator_object_id")
    _require(route.get("objects") == ["airplane"],
             "route must contain only airplane")
    _require(route.get("base_expert") == "source_e260",
             "base expert must be source_e260")
    _require(route.get("object_route") == {"airplane": "source_e260"},
             "airplane route must remain source_e260")
    _require(route.get("provenance", {}).get("official_actor_checkpoint") is None,
             "official actor checkpoint is forbidden")

    expected_experts = collector.get("candidate_experts")
    _require(isinstance(expected_experts, list) and len(expected_experts) == 6,
             "collector must declare exactly six candidate experts")
    _require(len(set(expected_experts)) == 6, "candidate experts must be unique")
    _require(route.get("candidate_experts") == expected_experts,
             "candidate expert order differs from canonical route")
    _require(list(route.get("experts", {})) == expected_experts,
             "expert checkpoint order differs from candidate order")

    temporal = collector.get("temporal_contract", {})
    _require(temporal.get("history_steps") == 10, "history must be 10 steps")
    _require(temporal.get("option_steps") == 10, "option must be 10 steps")
    _require(temporal.get("future_steps") == 20, "future must be 20 steps")
    _require(temporal.get("trigger") ==
             "first_valid_hand_object_contact_after_complete_history",
             "trigger contract differs")
    _require(temporal.get("first_episode_only") is True,
             "collection must be first-episode only")
    _require(temporal.get("post_option_policy") == "canonical_route_expert",
             "post-option policy must be the canonical route expert")

    assignment = collector.get("assignment", {})
    _require(assignment.get("scheme") == "seeded_balanced_per_environment",
             "assignment scheme differs")
    _require(assignment.get("arms") == 6, "assignment must have six arms")
    _require(assignment.get("propensity_numerator") == 1 and
             assignment.get("propensity_denominator") == 6,
             "assignment propensity must be 1/6")

    splits = collector.get("collections", {})
    _require(set(splits) == {"fit", "holdout"},
             "collector must own exactly fit and holdout splits")
    expected_seeds = {
        "fit": (256, 20260928256, 30),
        "holdout": (257, 20260928257, 30),
    }
    for name, (sim_seed, assignment_seed, minimum) in expected_seeds.items():
        split = splits[name]
        _require(split.get("simulator_seed") == sim_seed,
                 f"{name} simulator seed differs")
        _require(split.get("assignment_seed") == assignment_seed,
                 f"{name} assignment seed differs")
        _require(split.get("minimum_valid_rows_per_arm") == minimum,
                 f"{name} row minimum differs")
        num_envs = split.get("num_envs")
        _require(isinstance(num_envs, int) and num_envs > 0 and num_envs % 6 == 0,
                 f"{name} num_envs must be positive and divisible by six")

    motion_names = [motion.get("name") for motion in route.get("motions", [])]
    expected_motions = collector.get("motions")
    _require(expected_motions == [
        "s3_airplane_lift", "s7_airplane_lift_Retake", "s9_airplane_lift"],
        "collector must declare the three canonical airplane motions")
    _require(motion_names == expected_motions,
             "motion order differs from canonical route")
    motion_root = collector.get("motion_root")
    _require(motion_root == "outputs/CmResidual/agent_contact_option_airplane_motions",
             "motion root differs")
    for motion in route["motions"]:
        _require(str(Path(motion["path"]).parent) == motion_root,
                 f"motion path escapes canonical root: {motion['name']}")
        _require(len(motion.get("interaction_hand_sha256", "")) == 64,
                 f"motion hash missing: {motion['name']}")

    required_fields = {
        "env_id", "motion_id", "motion_name", "object_name",
        "simulator_object_id", "route_expert", "assignment",
        "assignment_propensity", "trigger_step", "start_frame", "state",
        "base_action", "candidate_actions", "history_state",
        "history_action", "history_contact", "option_candidate_action",
        "option_executed_action", "future_contact_mask",
        "future_contact_supported_lift_m", "followup_contact_fraction",
        "followup_max_contact_lift_m", "final_lift_success",
        "final_max_contact_lift_m", "final_contact_fraction",
        "final_episode_steps",
        "episode_id", "split", "pre_action_observation",
        "object_pose_t_object_local_frame",
        "object_pose_t_plus_1_object_local_frame",
        "target_delta_object_local_1",
        "contact_mask_t_plus_1_to_t_plus_5",
        "candidate_expert_names_and_checkpoint_sha256",
        "executed_action", "router_teacher_candidate_id",
        "router_teacher_action", "router_model_sha256",
        "router_input_state_sha256", "router_teacher_source",
        "route_config_sha256", "collector_config_sha256",
    }
    _require(set(collector.get("record_schema", {})) == required_fields,
             "record schema fields differ from the Probe card contract")

    checkpoint_hashes = {
        name: route["experts"][name]["sha256"] for name in expected_experts
    }
    motion_hashes = {
        motion["name"]: motion["interaction_hand_sha256"]
        for motion in route["motions"]
    }
    artifact_sizes: Dict[str, int] = {}
    if verify_artifacts:
        for name in expected_experts:
            spec = route["experts"][name]
            path = (artifact_root / spec["checkpoint"]).resolve()
            _require(path.is_file(), f"missing expert checkpoint: {name}")
            _require(sha256_path(path) == spec["sha256"],
                     f"expert checkpoint hash drift: {name}")
            artifact_sizes[f"expert:{name}"] = path.stat().st_size
        for motion in route["motions"]:
            path = (artifact_root / motion["path"] /
                    "interaction_hand_inspire.pt").resolve()
            _require(path.is_file(), f"missing motion tensor: {motion['name']}")
            _require(sha256_path(path) == motion["interaction_hand_sha256"],
                     f"motion tensor hash drift: {motion['name']}")
            artifact_sizes[f"motion:{motion['name']}"] = path.stat().st_size

    return {
        "collector": collector,
        "route": route,
        "collector_config_sha256": sha256_path(collector_path),
        "route_config_sha256": route_sha,
        "expert_checkpoint_sha256": checkpoint_hashes,
        "motion_sha256": motion_hashes,
        "artifact_sizes": artifact_sizes,
    }


def balanced_environment_assignment(num_envs: int, arms: int, seed: int) -> torch.Tensor:
    """Return a seeded block randomization with marginal propensity 1/arms."""
    if num_envs <= 0 or arms < 2 or num_envs % arms:
        raise ContractError("num_envs must be positive and divisible by arms")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    labels = torch.arange(num_envs, dtype=torch.int64) % arms
    return labels[torch.randperm(num_envs, generator=generator)]


def eligible_trigger_mask(
        contact: torch.Tensor,
        triggered: torch.Tensor,
        episode_ended: torch.Tensor,
        reset: torch.Tensor,
        progress: torch.Tensor,
        history_count: torch.Tensor,
        history_steps: int = 10) -> torch.Tensor:
    shape = contact.shape
    if any(value.shape != shape for value in
           (triggered, episode_ended, reset, progress, history_count)):
        raise ContractError("trigger tensors must share one-dimensional shape")
    if contact.ndim != 1:
        raise ContractError("trigger tensors must be one-dimensional")
    return (contact.bool() & ~triggered.bool() & ~episode_ended.bool() &
            ~reset.bool() & progress.gt(0) & history_count.ge(history_steps))


def option_active_mask(
        triggered: torch.Tensor,
        episode_ended: torch.Tensor,
        option_elapsed: torch.Tensor,
        option_steps: int = 10) -> torch.Tensor:
    return (triggered.bool() & ~episode_ended.bool() &
            option_elapsed.lt(option_steps))


@torch.no_grad()
def record_future_step(
        *,
        triggered: torch.Tensor,
        episode_ended: torch.Tensor,
        future_valid: torch.Tensor,
        future_elapsed: torch.Tensor,
        future_contact_mask: torch.Tensor,
        future_contact_supported_lift_m: torch.Tensor,
        post_contact: torch.Tensor,
        lift_m: torch.Tensor,
        done: torch.Tensor,
        future_steps: int = 20) -> torch.Tensor:
    """Record one terminal-aware future step and return the active row mask.

    A terminal transition counts as a step from the first episode.  It is valid
    only when it is the twentieth future transition; termination before that
    invalidates the row and prevents any data from a reset episode entering it.
    """
    active = (triggered.bool() & ~episode_ended.bool() & future_valid.bool() &
              future_elapsed.lt(future_steps))
    rows = active.nonzero(as_tuple=False).reshape(-1)
    if rows.numel():
        slots = future_elapsed[rows].long()
        future_contact_mask[rows, slots] = post_contact[rows].bool()
        future_contact_supported_lift_m[rows, slots] = torch.where(
            post_contact[rows].bool(), lift_m[rows], torch.zeros_like(lift_m[rows]))
        future_elapsed[rows] += 1
        ended_too_soon = done[rows].bool() & future_elapsed[rows].lt(future_steps)
        if ended_too_soon.any():
            future_valid[rows[ended_too_soon]] = False
    return active


def _tensor(records: Mapping[str, object], name: str) -> torch.Tensor:
    value = records.get(name)
    if not isinstance(value, torch.Tensor):
        raise ContractError(f"record field is not a tensor: {name}")
    return value


def validate_record_payload(
        payload: Mapping[str, object],
        collector: Mapping[str, object],
        expected_provenance: Mapping[str, object] = None) -> dict:
    """Validate a completed saved payload before it is admitted as Probe data."""
    _require(payload.get("schema") == RECORD_SCHEMA, "record payload schema differs")
    _require(payload.get("run_status") == "COMPLETED", "record payload is incomplete")
    experts = list(collector["candidate_experts"])
    temporal = collector["temporal_contract"]
    _require(payload.get("candidate_experts") == experts,
             "payload expert order differs")
    _require(payload.get("base_expert") == "source_e260",
             "payload base expert differs")
    _require(payload.get("post_option_policy") == temporal["post_option_policy"],
             "payload post-option policy differs")
    _require(payload.get("history_steps") == temporal["history_steps"] and
             payload.get("option_steps") == temporal["option_steps"] and
             payload.get("future_steps") == temporal["future_steps"],
             "payload temporal horizons differ")
    provenance = payload.get("provenance")
    _require(isinstance(provenance, Mapping), "payload provenance missing")
    route_sha = collector["canonical_route"]["sha256"]
    _require(provenance.get("route_config_sha256") == route_sha and
             provenance.get("canonical_route_sha256") == route_sha,
             "payload canonical route hash differs")
    _require(isinstance(provenance.get("evaluator_sha256"), str) and
             len(provenance["evaluator_sha256"]) == 64,
             "payload evaluator SHA256 missing")
    _require(isinstance(provenance.get("evaluator_git_blob_sha1"), str) and
             len(provenance["evaluator_git_blob_sha1"]) == 40,
             "payload evaluator blob hash missing")
    if expected_provenance is not None:
        _require(provenance.get("collector_config_sha256") ==
                 expected_provenance.get("collector_config_sha256"),
                 "payload collector config hash differs")
        _require(provenance.get("expert_checkpoint_sha256") ==
                 expected_provenance.get("expert_checkpoint_sha256"),
                 "payload expert checkpoint hashes differ")
        _require(provenance.get("motion_sha256") ==
                 expected_provenance.get("motion_sha256"),
                 "payload motion hashes differ")
    records = payload.get("records")
    _require(isinstance(records, Mapping), "payload records missing")

    env_id = _tensor(records, "env_id")
    n = int(env_id.numel())
    _require(env_id.shape == (n,), "env_id shape differs")
    shapes = {
        "motion_id": (n,), "simulator_object_id": (n,), "assignment": (n,),
        "assignment_propensity": (n,), "trigger_step": (n,), "start_frame": (n,),
        "state": (n, STATE_DIM), "base_action": (n, ACTION_DIM),
        "candidate_actions": (n, len(experts), ACTION_DIM),
        "history_state": (n, 10, STATE_DIM),
        "history_action": (n, 10, ACTION_DIM), "history_contact": (n, 10),
        "option_candidate_action": (n, 10, ACTION_DIM),
        "option_executed_action": (n, 10, ACTION_DIM),
        "future_contact_mask": (n, 20),
        "future_contact_supported_lift_m": (n, 20),
        "pre_action_observation": (n, 1442),
        "object_pose_t_object_local_frame": (n, 3),
        "object_pose_t_plus_1_object_local_frame": (n, 3),
        "target_delta_object_local_1": (n, 3),
        "contact_mask_t_plus_1_to_t_plus_5": (n, 5),
        "executed_action": (n, ACTION_DIM),
        "router_teacher_candidate_id": (n,),
        "router_teacher_action": (n, ACTION_DIM),
        "followup_contact_fraction": (n,), "followup_max_contact_lift_m": (n,),
        "final_lift_success": (n,), "final_max_contact_lift_m": (n,),
        "final_contact_fraction": (n,), "final_episode_steps": (n,),
    }
    for name, shape in shapes.items():
        _require(tuple(_tensor(records, name).shape) == shape,
                 f"record shape differs: {name}")
    for name in ("motion_name", "object_name", "route_expert", "episode_id", "split",
                 "router_model_sha256", "router_input_state_sha256",
                 "router_teacher_source", "route_config_sha256", "collector_config_sha256"):
        value = records.get(name)
        _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and
                 len(value) == n, f"record string field differs: {name}")
    _require(all(value in {"fit", "holdout"} for value in records["split"]),
             "record split differs")
    _require(all(isinstance(value, str) and value for value in records["episode_id"]),
             "record episode_id differs")
    _require(len(set(records["episode_id"])) == n,
             "episode_id values must be unique within the run")
    _require(all(value == "c1_observation_router" for value in records["router_teacher_source"]),
             "router teacher fallback is forbidden")
    for name in ("history_contact", "future_contact_mask", "contact_mask_t_plus_1_to_t_plus_5",
                 "final_lift_success"):
        _require(_tensor(records, name).dtype == torch.bool,
                 f"boolean record field has wrong dtype: {name}")
    motion_ids = _tensor(records, "motion_id").long()
    _require(bool(((motion_ids >= 0) &
                   (motion_ids < len(collector["motions"]))).all()),
             "motion id outside canonical route")
    _require(all(records["motion_name"][index] ==
                 collector["motions"][int(motion_ids[index])]
                 for index in range(n)), "motion id/name mismatch")
    _require(bool((_tensor(records, "simulator_object_id") >= 0).all()),
             "negative simulator object id")

    assignment = _tensor(records, "assignment").long()
    _require(bool(((assignment >= 0) & (assignment < len(experts))).all()),
             "assignment outside six arms")
    propensity = _tensor(records, "assignment_propensity").float()
    _require(bool(torch.allclose(propensity,
                                 torch.full_like(propensity, 1.0 / len(experts)),
                                 rtol=0.0, atol=1e-7)),
             "assignment propensity differs from 1/6")
    _require(all(value == "airplane" for value in records["object_name"]),
             "payload contains a non-airplane object")
    _require(all(value == "source_e260" for value in records["route_expert"]),
             "payload route expert differs")
    _require(set(records["motion_name"]).issubset(set(collector["motions"])),
             "payload contains an unknown motion")

    finite_names = [
        "state", "base_action", "candidate_actions", "history_state",
        "history_action", "option_candidate_action", "option_executed_action",
        "pre_action_observation", "object_pose_t_object_local_frame",
        "object_pose_t_plus_1_object_local_frame", "target_delta_object_local_1",
        "executed_action", "router_teacher_action",
        "future_contact_supported_lift_m", "followup_contact_fraction",
        "followup_max_contact_lift_m", "final_max_contact_lift_m",
        "final_contact_fraction",
    ]
    for name in finite_names:
        _require(bool(torch.isfinite(_tensor(records, name).float()).all()),
                 f"nonfinite record field: {name}")

    option_candidate = _tensor(records, "option_candidate_action")
    option_executed = _tensor(records, "option_executed_action")
    _require(bool(torch.allclose(option_candidate, option_executed,
                                 rtol=0.0, atol=1e-7)),
             "assigned option was not executed for all ten steps")
    if n:
        rows = torch.arange(n)
        at_trigger = _tensor(records, "candidate_actions")[rows, assignment]
        _require(bool(torch.allclose(at_trigger, option_candidate[:, 0],
                                     rtol=0.0, atol=1e-7)),
                 "first option action differs from the assigned trigger candidate")

    _require(bool(torch.allclose(
        _tensor(records, "executed_action"), option_executed[:, 0],
        rtol=0.0, atol=1e-7)), "executed action differs from trigger option action")
    teacher_id = _tensor(records, "router_teacher_candidate_id").long()
    _require(bool(((teacher_id >= 0) & (teacher_id < len(experts))).all()),
             "router teacher candidate outside six arms")
    teacher_action = _tensor(records, "router_teacher_action")
    candidates = _tensor(records, "candidate_actions")
    rows = torch.arange(n)
    _require(bool(torch.allclose(teacher_action, candidates[rows, teacher_id],
                                 rtol=0.0, atol=1e-7)),
             "router teacher action differs from its candidate")
    _require(bool(torch.allclose(
        _tensor(records, "target_delta_object_local_1"),
        _tensor(records, "object_pose_t_plus_1_object_local_frame") -
        _tensor(records, "object_pose_t_object_local_frame"),
        rtol=0.0, atol=1e-9)),
        "object-local signed delta differs from pose pair")
    _require(bool(torch.allclose(
        _tensor(records, "contact_mask_t_plus_1_to_t_plus_5").float(),
        _tensor(records, "future_contact_mask")[:, :5].float(),
        rtol=0.0, atol=0.0)),
        "five-step contact target differs from future contact mask")

    contact = _tensor(records, "future_contact_mask").bool()
    supported = _tensor(records, "future_contact_supported_lift_m").float()
    _require(bool((supported[~contact] == 0).all()),
             "future supported lift must be zero outside contact")
    expected_fraction = contact.float().mean(dim=1)
    _require(bool(torch.allclose(_tensor(records, "followup_contact_fraction").float(),
                                 expected_fraction, rtol=0.0, atol=1e-7)),
             "future contact aggregate differs")
    expected_max = supported.max(dim=1).values.clamp_min(0) if n else supported.new_empty(0)
    _require(bool(torch.allclose(_tensor(records, "followup_max_contact_lift_m").float(),
                                 expected_max, rtol=0.0, atol=1e-7)),
             "future lift aggregate differs")
    _require(bool((_tensor(records, "start_frame") >= 0).all()),
             "negative start frame")
    _require(bool((_tensor(records, "final_episode_steps") >=
                   _tensor(records, "trigger_step") + 20).all()),
             "future labels cross the first-episode boundary")
    return {
        "rows": n,
        "arm_counts": {
            experts[index]: int((assignment == index).sum())
            for index in range(len(experts))
        },
    }
