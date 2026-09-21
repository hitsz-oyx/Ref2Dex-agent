"""Shared, policy-independent approach shaping for matched scratch PPO arms."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ApproachConfig:
    point_stride: int = 8
    # Observed scratch states start around 0.78 m apart. A 1 m linear
    # potential retains a useful slope there and is bounded in [0, 1].
    proximity_scale_m: float = 1.0


@torch.inference_mode()
def sampled_surface_gap(hand_points: torch.Tensor, object_points: torch.Tensor,
                        config: ApproachConfig = ApproachConfig()) -> torch.Tensor:
    """Minimum sampled hand/object surface separation for each environment."""
    if (config.point_stride < 1 or config.proximity_scale_m <= 0 or
            hand_points.ndim != 3 or object_points.ndim != 3 or
            hand_points.shape[0] != object_points.shape[0] or
            hand_points.shape[-1] != 3 or object_points.shape[-1] != 3 or
            not hand_points.shape[1] or not object_points.shape[1]):
        raise ValueError("invalid approach geometry or configuration")
    if not torch.isfinite(hand_points).all() or not torch.isfinite(object_points).all():
        raise FloatingPointError("approach geometry must be finite")
    sampled_hand = hand_points[:, ::config.point_stride].contiguous()
    sampled_object = object_points[:, ::config.point_stride].contiguous()
    return torch.cdist(sampled_hand, sampled_object).amin(dim=(1, 2))


@torch.inference_mode()
def potential_approach_reward(gap_before: torch.Tensor, gap_after: torch.Tensor,
                              done: torch.Tensor, *, gamma: float,
                              config: ApproachConfig = ApproachConfig()) -> torch.Tensor:
    """Potential shaping gamma*Phi(s')-Phi(s), with Phi(terminal)=0.

    Positive proximity potential penalizes losing proximity and does not reward
    indefinite hovering; environment terminations cannot pick up a reset gap.
    """
    if (gap_before.ndim != 1 or gap_after.shape != gap_before.shape or
            done.shape != gap_before.shape or done.dtype != torch.bool or
            not (0 < gamma <= 1) or config.proximity_scale_m <= 0):
        raise ValueError("invalid approach reward inputs")
    if (not torch.isfinite(gap_before).all() or not torch.isfinite(gap_after).all() or
            (gap_before < 0).any() or (gap_after < 0).any()):
        raise FloatingPointError("approach gaps must be nonnegative and finite")
    before = (1.0 - gap_before / config.proximity_scale_m).clamp(0.0, 1.0)
    after = (1.0 - gap_after / config.proximity_scale_m).clamp(0.0, 1.0)
    return gamma * torch.where(done, torch.zeros_like(after), after) - before
