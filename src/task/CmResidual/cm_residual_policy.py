"""Bounded object-frame residual Cm and policy contracts.

The module is deliberately independent of Isaac Gym.  Native collectors provide
the current state and baseline absolute PD target; this module only converts a
small object-frame translation residual, predicts short consequences, and
returns either that residual or an exact zero fallback.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


RESIDUAL_SCALE_M = (0.002, 0.002, 0.003)
HISTORY_DIM = 69
CONTEXT_DIM = 38
TARGET_DIM = 6  # local dx/dy/dz in mm, joint contact, contact loss, clearance loss
SCHEMA = "ref2dex.cm_residual_policy.v2"
VALID_POLICY_MODES = ("cm",)


def _finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values")


def _quat_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.ndim != 2 or quaternion.shape[-1] != 4:
        raise ValueError("quaternion must be [B,4]")
    _finite("quaternion", quaternion)
    q = torch.nn.functional.normalize(quaternion, dim=-1, eps=1e-8)
    x, y, z, w = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), -1).reshape(-1, 3, 3)


def local_to_world(quaternion: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """Rotate object-local vectors into world coordinates."""
    if local.ndim != 2 or local.shape != (len(quaternion), 3):
        raise ValueError("local vectors must be [B,3]")
    _finite("local", local)
    return torch.einsum("bij,bj->bi", _quat_matrix(quaternion), local)


def world_to_local(quaternion: torch.Tensor, world: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors into the current object frame."""
    if world.ndim != 2 or world.shape != (len(quaternion), 3):
        raise ValueError("world vectors must be [B,3]")
    _finite("world", world)
    return torch.einsum("bij,bj->bi", _quat_matrix(quaternion).transpose(-1, -2), world)


def world_height_score_mm(object_quaternion: torch.Tensor, local_delta_mm: torch.Tensor) -> torch.Tensor:
    """Return the world-frame z displacement used by the native utility metric."""
    if object_quaternion.ndim != 2 or object_quaternion.shape[-1] != 4:
        raise ValueError("object_quaternion must be [B,4]")
    if local_delta_mm.ndim != 2 or local_delta_mm.shape != (len(object_quaternion), 3):
        raise ValueError("local_delta_mm must be [B,3]")
    return local_to_world(object_quaternion, local_delta_mm)[:, 2]


def validate_policy_mode(mode: str) -> None:
    """Reject the old candidate-order permutation masquerading as a control."""
    if mode not in VALID_POLICY_MODES:
        raise ValueError(
            f"{mode!r} is not a valid residual policy control; shuffled candidate order "
            "does not define an independent decision policy"
        )


def residual_scale(device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(RESIDUAL_SCALE_M, device=device, dtype=dtype)


def bounded_residual(value: torch.Tensor) -> torch.Tensor:
    """Clamp normalized residual coordinates to the fixed physical authority."""
    if value.ndim != 2 or value.shape[-1] != 3:
        raise ValueError("normalized residual must be [B,3]")
    _finite("normalized residual", value)
    return value.clamp(-1.0, 1.0) * residual_scale(value.device, value.dtype)


def compose_residual_target(
    baseline_target: torch.Tensor,
    object_pose: torch.Tensor,
    residual_local_m: torch.Tensor,
    *,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Add a bounded local translation to an absolute native PD target.

    Only native translation DOFs 0:3 may change.  A zero residual returns an
    exact clone of the baseline target, which makes the baseline arm auditable.
    """
    if baseline_target.ndim != 2 or baseline_target.shape[-1] != 18:
        raise ValueError("baseline_target must be [B,18]")
    if object_pose.ndim != 2 or object_pose.shape[-1] != 7 or len(object_pose) != len(baseline_target):
        raise ValueError("object_pose must be [B,7]")
    if residual_local_m.shape != (len(baseline_target), 3):
        raise ValueError("residual_local_m must be [B,3]")
    _finite("baseline target", baseline_target)
    _finite("object pose", object_pose)
    _finite("residual", residual_local_m)
    scale = residual_scale(residual_local_m.device, residual_local_m.dtype)
    if (residual_local_m.abs() > scale + 1e-7).any():
        raise ValueError("residual exceeds configured object-frame authority")
    world_delta = local_to_world(object_pose[:, 3:7], residual_local_m)
    requested = torch.zeros_like(baseline_target)
    requested[:, :3] = world_delta
    target = baseline_target.clone()
    active = residual_local_m.abs().amax(-1) > 0
    if active.any():
        target[active, :3] += world_delta[active]
        if lower is not None or upper is not None:
            if lower is None or upper is None or lower.shape != (18,) or upper.shape != (18,):
                raise ValueError("lower and upper must both be [18]")
            target[active] = target[active].clamp(lower, upper)
    applied = target - baseline_target
    return target, {
        "requested_delta_world": requested,
        "applied_delta_world": applied,
        "active": active,
        "saturated": (applied[:, :3] - world_delta).abs().amax(-1) > 1e-7,
    }


def apply_residual_action(
    baseline_action: torch.Tensor,
    object_quaternion: torch.Tensor,
    residual_local_m: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the same translation residual in DExplore normalized action space.

    DExplore's first three normalized action coordinates are world translation
    offsets in metres, so adding the rotated delta is exactly equivalent to
    adding it to the native baseline PD target for those coordinates.
    """
    if baseline_action.ndim != 2 or baseline_action.shape[-1] != 18:
        raise ValueError("baseline_action must be [B,18]")
    if residual_local_m.shape != (len(baseline_action), 3):
        raise ValueError("residual shape mismatch")
    scale = residual_scale(residual_local_m.device, residual_local_m.dtype)
    if (residual_local_m.abs() > scale + 1e-7).any():
        raise ValueError("residual exceeds configured object-frame authority")
    world_delta = local_to_world(object_quaternion, residual_local_m)
    proposed = baseline_action.clone()
    proposed[:, :3] += world_delta
    applied = proposed[:, :3].clamp(-1.0, 1.0) - baseline_action[:, :3]
    saturated = (applied - world_delta).abs().amax(-1) > 1e-7
    result = baseline_action.clone()
    result[:, :3] += applied
    return result, {"requested_delta_world": world_delta, "applied_delta_world": applied,
                    "saturated": saturated}


def context_features(
    state: torch.Tensor,
    hand_force: torch.Tensor,
    object_force: torch.Tensor,
    mass: torch.Tensor,
    gravity: float | torch.Tensor,
    clearance: torch.Tensor,
    rest_z: torch.Tensor,
) -> torch.Tensor:
    """Build fixed current-only physical context for Cm and the actor."""
    if state.ndim != 2 or state.shape[-1] != 49:
        raise ValueError("state must be [B,49]")
    n = len(state)
    if hand_force.ndim != 3 or hand_force.shape[0] != n or hand_force.shape[-1] != 3:
        raise ValueError("hand_force must be [B,L,3]")
    if object_force.shape != (n, 3) or mass.shape != (n,) or clearance.shape != (n,) or rest_z.shape != (n,):
        raise ValueError("physical context shapes do not match state")
    _finite("state", state)
    weight = mass * torch.as_tensor(gravity, device=state.device, dtype=state.dtype)
    if (weight <= 0).any():
        raise ValueError("mass*gravity must be positive")
    hand_norm = hand_force.norm(dim=-1).amax(-1)
    object_norm = object_force.norm(dim=-1)
    log_force = torch.log1p(torch.stack((hand_norm, object_norm), -1) / weight[:, None])
    relative_position = state[:, 36:39] - state[:, :3]
    relative_velocity = state[:, 43:46] - state[:, 18:21]
    height = (state[:, 38] - rest_z).clamp_min(0)
    quaternion = torch.nn.functional.normalize(state[:, 39:43], dim=-1, eps=1e-8)
    result = torch.cat((
        state[:, :18], relative_position, relative_velocity,
        state[:, 43:49], quaternion, log_force,
        clearance[:, None], height[:, None],
    ), -1)
    if result.shape[-1] != CONTEXT_DIM:
        raise AssertionError(f"context dimension changed: {result.shape[-1]}")
    _finite("context", result)
    return result


def two_step_targets(
    state: torch.Tensor,
    future_state: torch.Tensor,
    future_contact: torch.Tensor,
    future_clearance: torch.Tensor,
    rest_z: torch.Tensor,
    *,
    step_index: int = 1,
) -> torch.Tensor:
    """Convert actual two-step outcomes to Cm targets.

    The first three values are object-local displacement in millimetres.  The
    remaining values are factual event labels, never online inputs.
    """
    if future_state.ndim != 3 or future_state.shape[-1] != 49 or len(future_state) != len(state):
        raise ValueError("future_state must be [B,H,49]")
    if not 0 <= step_index < future_state.shape[1]:
        raise ValueError("invalid future step")
    if future_contact.ndim != 3 or future_contact.shape[:2] != future_state.shape[:2] or future_contact.shape[-1] != 2:
        raise ValueError("future_contact must be [B,H,2]")
    if future_clearance.shape != future_state.shape[:2]:
        raise ValueError("future_clearance shape mismatch")
    pose_quat = state[:, 39:43]
    world_delta = future_state[:, step_index, 36:39] - state[:, 36:39]
    local_delta = world_to_local(pose_quat, world_delta) * 1000.0
    contact = future_contact[:, step_index].all(-1).float()
    contact_loss = (~future_contact[:, : step_index + 1].all(-1).all(-1)).float()
    clearance_loss = (future_clearance[:, : step_index + 1] < 0.002).any(-1).float()
    return torch.cat((local_delta, contact[:, None], contact_loss[:, None], clearance_loss[:, None]), -1)


class ResidualConsequenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.history = nn.GRU(HISTORY_DIM, 64, batch_first=True)
        self.context = nn.Sequential(nn.Linear(CONTEXT_DIM, 64), nn.SiLU(), nn.LayerNorm(64))
        self.residual = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.LayerNorm(32))
        self.head = nn.Sequential(nn.Linear(160, 128), nn.SiLU(), nn.Linear(128, TARGET_DIM))

    def forward(self, history: torch.Tensor, context: torch.Tensor,
                residual_normalized: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != HISTORY_DIM:
            raise ValueError("history must be [B,10,69]")
        if context.shape != (len(history), CONTEXT_DIM):
            raise ValueError("context shape mismatch")
        if residual_normalized.shape != (len(history), 3):
            raise ValueError("residual shape mismatch")
        _, hidden = self.history(history)
        return self.head(torch.cat((hidden[-1], self.context(context), self.residual(residual_normalized)), -1))


class ResidualActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.history = nn.GRU(HISTORY_DIM, 64, batch_first=True)
        self.context = nn.Sequential(nn.Linear(CONTEXT_DIM, 64), nn.SiLU(), nn.LayerNorm(64))
        self.head = nn.Sequential(nn.Linear(128, 96), nn.SiLU(), nn.Linear(96, 3))

    def forward(self, history: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        _, hidden = self.history(history)
        return self.head(torch.cat((hidden[-1], self.context(context)), -1)).tanh()


def consequence_score(
    output: torch.Tensor, object_quaternion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if output.ndim != 2 or output.shape[-1] != TARGET_DIM:
        raise ValueError("consequence output shape mismatch")
    if object_quaternion.shape != (len(output), 4):
        raise ValueError("object_quaternion shape mismatch")
    joint = output[:, 3].sigmoid()
    contact_loss = output[:, 4].sigmoid()
    clearance_loss = output[:, 5].sigmoid()
    score_mm = world_height_score_mm(object_quaternion, output[:, :3]) * joint * (1.0 - clearance_loss)
    return score_mm, contact_loss, clearance_loss


class FrozenResidualPolicy:
    """Frozen Cm+actor inference with fail-closed baseline fallback."""

    def __init__(self, checkpoint: str | Path, device: torch.device | str, mode: str = "cm") -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != SCHEMA:
            raise ValueError("residual policy schema mismatch")
        validate_policy_mode(mode)
        self.device = torch.device(device)
        self.mode = mode
        self.history_mean = payload["history_mean"].to(self.device)
        self.history_std = payload["history_std"].to(self.device)
        self.context_mean = payload["context_mean"].to(self.device)
        self.context_std = payload["context_std"].to(self.device)
        self.margin_mm = float(payload["margins_mm"][mode])
        self.uncertainty_mm = float(payload["uncertainty_mm"][mode])
        self.models = []
        if set(payload["models"]) != set(VALID_POLICY_MODES):
            raise ValueError("residual checkpoint must contain only an independent Cm policy")
        for state in payload["models"][mode]:
            model = ResidualConsequenceModel().to(self.device)
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False)
            self.models.append(model)
        self.actor = ResidualActor().to(self.device)
        self.actor.load_state_dict(payload["actors"][mode], strict=True)
        self.actor.eval().requires_grad_(False)

    @torch.no_grad()
    def choose(
        self, history: torch.Tensor, context: torch.Tensor, object_quaternion: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if object_quaternion.shape != (len(history), 4):
            raise ValueError("object_quaternion shape mismatch")
        h_unscaled = (history - self.history_mean) / self.history_std
        c_unscaled = (context - self.context_mean) / self.context_std
        ood = (h_unscaled.abs() > 8).flatten(1).any(-1) | (c_unscaled.abs() > 8).any(-1)
        h = h_unscaled.clamp(-8, 8)
        c = c_unscaled.clamp(-8, 8)
        normalized = self.actor(h, c)
        candidate = normalized * residual_scale(self.device, normalized.dtype)
        zero = torch.zeros_like(normalized)
        predictions = []
        zero_predictions = []
        for model in self.models:
            predictions.append(model(h, c, normalized))
            zero_predictions.append(model(h, c, zero))
        predicted = torch.stack(predictions)
        zero_predicted = torch.stack(zero_predictions)
        quaternion = object_quaternion[None].expand(len(self.models), -1, -1).reshape(-1, 4)
        score, loss, clearance_loss = consequence_score(predicted.flatten(0, 1), quaternion)
        zero_score, zero_loss, zero_clearance = consequence_score(zero_predicted.flatten(0, 1), quaternion)
        score = score.view(len(self.models), -1)
        loss = loss.view(len(self.models), -1)
        clearance_loss = clearance_loss.view(len(self.models), -1)
        zero_score = zero_score.view(len(self.models), -1)
        zero_loss = zero_loss.view(len(self.models), -1)
        zero_clearance = zero_clearance.view(len(self.models), -1)
        mean_score = score.mean(0)
        std_score = score.std(0, unbiased=False)
        accepted = (~ood & (mean_score - zero_score.mean(0) > self.margin_mm)
                    & (std_score <= self.uncertainty_mm)
                    & (loss.mean(0) <= zero_loss.mean(0) + .02)
                    & (clearance_loss.mean(0) <= zero_clearance.mean(0) + .02))
        output = torch.where(accepted[:, None], candidate, torch.zeros_like(candidate))
        return output, dict(
            proposed=candidate, accepted=accepted, ood=ood, score_mm=mean_score,
            zero_score_mm=zero_score.mean(0), score_std_mm=std_score,
            contact_loss=loss.mean(0), zero_contact_loss=zero_loss.mean(0),
            clearance_loss=clearance_loss.mean(0), zero_clearance_loss=zero_clearance.mean(0),
        )
