"""PointWAM trajectory loss from the published training recipe.

Squared error is averaged per coordinate. Motion weights use adjacent-frame
displacement and stay positive for static points. This is the supervision
term only; it does not include the robot-action term.
"""

from __future__ import annotations

import torch


def motion_weights(positions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Normalize paper motion weights over valid point-step pairs.

    ``positions`` is ``[B, 31, N, 3]`` and ``valid`` is ``[B, 30, N]``.
    """
    if positions.ndim != 4 or positions.shape[-1] != 3 or positions.shape[1] != valid.shape[1] + 1:
        raise ValueError("positions must cover one more frame than the target mask")
    adjacent = torch.isfinite(positions[:, 1:]).all(dim=-1) & torch.isfinite(positions[:, :-1]).all(dim=-1)
    delta = torch.linalg.norm(positions[:, 1:] - positions[:, :-1], dim=-1)
    raw = torch.sigmoid(5 * (delta / 0.005 - 1))
    raw = torch.where(valid & adjacent, raw, torch.zeros_like(raw))
    return raw / raw.sum().clamp_min(1e-8)


def standardized_joint_loss(
    scene_prediction: torch.Tensor,
    hand_prediction: torch.Tensor,
    scene_target: torch.Tensor,
    hand_target: torch.Tensor,
    scene_valid: torch.Tensor,
    hand_valid: torch.Tensor,
    scene_positions: torch.Tensor,
    hand_positions: torch.Tensor,
    stats,
) -> torch.Tensor:
    """Paper loss: equal scene and hand terms on standardized displacements.

    Motion weights still use metric positions. ``stats`` is a ``HorizonStats``.
    Predictions are already standardized and use time on axis 0.
    """
    scene_std = stats.standardize_scene(scene_target)
    hand_std = stats.standardize_hand(hand_target)
    scene_loss = trajectory_loss(
        scene_prediction.unsqueeze(0), scene_std.unsqueeze(0), scene_valid.unsqueeze(0), scene_positions.unsqueeze(0))
    flat_hand_prediction = hand_prediction.reshape(hand_prediction.shape[0], 20, 3)
    flat_hand_target = hand_std.reshape(hand_std.shape[0], 20, 3)
    flat_hand_valid = hand_valid.reshape(hand_valid.shape[0], 20)
    flat_hand_positions = hand_positions.reshape(hand_positions.shape[0], 20, 3)
    if bool(flat_hand_valid.any()):
        hand_loss = trajectory_loss(
            flat_hand_prediction.unsqueeze(0), flat_hand_target.unsqueeze(0),
            flat_hand_valid.unsqueeze(0), flat_hand_positions.unsqueeze(0))
    else:
        hand_loss = scene_loss.new_zeros(())
    return scene_loss + hand_loss


def trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean per-coordinate squared error on valid point-step pairs."""
    if prediction.shape != target.shape or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must share a trailing xyz axis")
    if valid.shape != prediction.shape[:-1]:
        raise ValueError("validity mask must match prediction without xyz")
    valid = valid & torch.isfinite(prediction).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    safe_prediction = torch.where(valid.unsqueeze(-1), prediction, torch.zeros_like(prediction))
    safe_target = torch.where(valid.unsqueeze(-1), target, torch.zeros_like(target))
    error = ((safe_prediction - safe_target) ** 2).sum(dim=-1) / 3
    if positions is None:
        weights = valid.to(error.dtype)
        weights = weights / weights.sum().clamp_min(1)
    else:
        weights = motion_weights(positions, valid)
    return (error * weights).sum()
