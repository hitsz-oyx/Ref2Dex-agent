"""Shared dense reward for lifting an object while retaining contact."""
from __future__ import annotations

import torch


@torch.inference_mode()
def held_lift_reward(object_z: torch.Tensor, rest_z: torch.Tensor,
                     hand_contact: torch.Tensor, object_contact: torch.Tensor,
                     *, target_height_m: float = 0.03) -> torch.Tensor:
    """Return a bounded [0, 1] reward for contact-supported lift height."""
    if (object_z.ndim != 1 or rest_z.shape != object_z.shape or
            hand_contact.shape != object_z.shape or object_contact.shape != object_z.shape or
            hand_contact.dtype != torch.bool or object_contact.dtype != torch.bool or
            target_height_m <= 0):
        raise ValueError("invalid held-lift reward inputs")
    if not torch.isfinite(object_z).all() or not torch.isfinite(rest_z).all():
        raise FloatingPointError("held-lift heights must be finite")
    height = ((object_z - rest_z) / target_height_m).clamp(0.0, 1.0)
    return height * (hand_contact & object_contact).to(height.dtype)


@torch.inference_mode()
def contact_lift_progress_reward(object_z_before: torch.Tensor,
                                 object_z_after: torch.Tensor,
                                 hand_contact: torch.Tensor,
                                 object_contact: torch.Tensor,
                                 done: torch.Tensor,
                                 *, target_step_m: float = 0.003) -> torch.Tensor:
    """Reward signed, contact-supported vertical progress in one control step."""
    if (object_z_before.ndim != 1 or object_z_after.shape != object_z_before.shape or
            hand_contact.shape != object_z_before.shape or
            object_contact.shape != object_z_before.shape or done.shape != object_z_before.shape or
            hand_contact.dtype != torch.bool or object_contact.dtype != torch.bool or
            done.dtype != torch.bool or target_step_m <= 0):
        raise ValueError("invalid contact lift-progress reward inputs")
    if not (torch.isfinite(object_z_before).all() and torch.isfinite(object_z_after).all()):
        raise FloatingPointError("lift-progress heights must be finite")
    progress = ((object_z_after - object_z_before) / target_step_m).clamp(-1.0, 1.0)
    valid = hand_contact & object_contact & ~done
    return progress * valid.to(progress.dtype)
