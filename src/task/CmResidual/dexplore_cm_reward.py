"""Frozen-Cmv2 dense reward for a from-scratch DExplore policy."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.task.CmResidual.dexplore_cm_geometry import DELTA_TIME_S, world_to_object_frame
from src.task.CmResidual.v121c_ranking import compose_current_local_delta


@dataclass(frozen=True)
class CmRewardConfig:
    effect_translation_scale: float = 1.0
    effect_translation_cap_m: float = 0.02
    position_sigma_m: float = 0.02
    improvement_temperature: float = 1.0
    interaction_object_chunk: int = 256
    compile_swept_topk: bool = True
    interaction_radius_m: float = 0.02
    prefilter_swept_sphere: bool = True
    positive_only: bool = False


@torch.inference_mode()
def swept_object_sphere_candidates(object_points: torch.Tensor, hand_points: torch.Tensor,
                                   hand_flow: torch.Tensor, *, radius_m: float = 0.02) -> torch.Tensor:
    """Conservative broad phase: false means no swept point can meet an object point.

    Every object point lies inside its enclosing sphere. For each hand point,
    compute its exact closest approach to that sphere's center along the
    nominal motion segment. If every segment misses the sphere plus interaction
    radius, Cmv2's exact swept point-pair distances cannot activate any token.
    """
    if (object_points.ndim != 3 or hand_points.ndim != 3 or
            hand_flow.shape != hand_points.shape or
            object_points.shape[0] != hand_points.shape[0] or
            object_points.shape[-1] != 3 or hand_points.shape[-1] != 3 or
            not object_points.shape[1] or not hand_points.shape[1] or
            not (0 < radius_m < float("inf"))):
        raise ValueError("invalid swept-sphere prefilter inputs")
    if not (torch.isfinite(object_points).all() and torch.isfinite(hand_points).all()
            and torch.isfinite(hand_flow).all()):
        raise FloatingPointError("swept-sphere geometry must be finite")
    center = object_points.mean(1)
    sphere_radius = (object_points - center[:, None]).norm(dim=-1).amax(-1)
    relative = hand_points - center[:, None]
    alpha = (-(relative * hand_flow).sum(-1) /
             hand_flow.square().sum(-1).clamp_min(1e-12)).clamp(0, 1)
    closest = (relative + alpha[..., None] * hand_flow).norm(dim=-1).amin(-1)
    # The extra margin protects the conservative guarantee from roundoff.
    return closest <= sphere_radius + radius_m + 1e-4


def bounded_translation_progress(current_pose: torch.Tensor, goal_pose: torch.Tensor,
                                 local_delta: torch.Tensor, valid: torch.Tensor,
                                 config: CmRewardConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Reward a predicted object translation that reduces next-reference error."""
    if current_pose.shape != goal_pose.shape or current_pose.shape[-2:] != (4, 4):
        raise ValueError("current_pose and goal_pose must match [B,4,4]")
    if local_delta.shape != (current_pose.shape[0], 6) or valid.shape != (current_pose.shape[0],):
        raise ValueError("local_delta/valid must be [B,6]/[B]")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    if (config.effect_translation_scale <= 0 or config.effect_translation_cap_m <= 0 or
            config.position_sigma_m <= 0 or config.improvement_temperature <= 0):
        raise ValueError("Cm reward scales must be positive")
    if not (torch.isfinite(current_pose).all() and torch.isfinite(goal_pose).all() and
            torch.isfinite(local_delta).all()):
        raise ValueError("Cm reward inputs must be finite")
    translation = local_delta[:, :3] * config.effect_translation_scale
    norm = translation.norm(dim=-1, keepdim=True)
    translation = translation * (config.effect_translation_cap_m / norm.clamp_min(
        config.effect_translation_cap_m)).clamp_max(1.0)
    calibrated = torch.cat((translation, torch.zeros_like(local_delta[:, 3:])), dim=-1)
    predicted_pose = compose_current_local_delta(current_pose, calibrated[:, None])[:, 0]
    denominator = config.position_sigma_m ** 2
    zero_cost = (current_pose[:, :3, 3] - goal_pose[:, :3, 3]).square().sum(-1) / denominator
    predicted_cost = (predicted_pose[:, :3, 3] - goal_pose[:, :3, 3]).square().sum(-1) / denominator
    improvement = zero_cost - predicted_cost
    reward = torch.tanh(improvement / config.improvement_temperature)
    if config.positive_only:
        reward = reward.clamp_min(0)
    return torch.where(valid, reward, torch.zeros_like(reward)), improvement


@torch.inference_mode()
def predict_cm_reward(*, adapter, bridge, action: torch.Tensor, current_native: torch.Tensor,
                      object_root_state: torch.Tensor, goal_pose: torch.Tensor,
                      lower: torch.Tensor, upper: torch.Tensor,
                      config: CmRewardConfig = CmRewardConfig()) -> dict[str, torch.Tensor]:
    """Evaluate only the policy's executed action; no expert action is supplied."""
    if action.ndim != 2 or action.shape[-1] != 18 or action.shape != current_native.shape:
        raise ValueError("action/current_native must match [B,18]")
    current = bridge.current(current_native, object_root_state)
    hand_points, hand_normals, hand_flow = bridge.nominal_hand_sweep(
        current_native, action[:, None], lower, upper)
    hand_flow = hand_flow[:, 0]
    candidates = (swept_object_sphere_candidates(
        current.object_points, hand_points, hand_flow,
        # The pinned frozen model uses 0.02 m. Never accept a narrower
        # configured radius, which could silently suppress real interactions.
        radius_m=max(0.02, config.interaction_radius_m)) if config.prefilter_swept_sphere else
        torch.ones(action.shape[0], dtype=torch.bool, device=action.device))
    reward = torch.zeros(action.shape[0], device=action.device, dtype=action.dtype)
    improvement = torch.zeros_like(reward)
    valid = torch.zeros_like(candidates)
    delta = torch.zeros((action.shape[0], 6), device=action.device, dtype=action.dtype)
    if not candidates.any():
        return {"reward": reward, "improvement": improvement, "valid": valid,
                "predicted_delta_xi": delta, "prefilter_candidate": candidates}
    selected = candidates.nonzero(as_tuple=True)[0]
    object_pose = current.object_pose.index_select(0, selected)
    object_points = world_to_object_frame(
        current.object_points.index_select(0, selected), object_pose, vector=False)
    object_normals = world_to_object_frame(
        current.object_normals.index_select(0, selected), object_pose, vector=True)
    hand_points = world_to_object_frame(hand_points.index_select(0, selected), object_pose, vector=False)
    hand_normals = world_to_object_frame(hand_normals.index_select(0, selected), object_pose, vector=True)
    hand_flow = world_to_object_frame(hand_flow.index_select(0, selected), object_pose, vector=True)
    output = adapter.predict_effect_only(
        object_points, object_normals, hand_points, hand_normals, hand_flow, DELTA_TIME_S,
        torch.ones((selected.numel(), 1538), dtype=torch.bool, device=action.device),
        interaction_object_chunk=config.interaction_object_chunk,
        compile_swept_topk=config.compile_swept_topk,
    )
    predicted_delta = output["delta_xi_root"]
    predicted_valid = (torch.isfinite(predicted_delta).all(-1) & output["token_mask"].any(-1) &
             torch.isfinite(output["token_mass"]).all(-1) & (output["token_mass"].sum(-1) > 0))
    selected_reward, selected_improvement = bounded_translation_progress(
        object_pose, goal_pose.index_select(0, selected), predicted_delta, predicted_valid, config)
    reward[selected] = selected_reward
    improvement[selected] = selected_improvement
    valid[selected] = predicted_valid
    delta[selected] = predicted_delta
    return {"reward": reward, "improvement": improvement, "valid": valid,
            "predicted_delta_xi": delta, "prefilter_candidate": candidates}
