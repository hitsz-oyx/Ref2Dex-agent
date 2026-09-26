"""Frozen Cm one-step effect as a normalized PPO critic salience prior."""
from __future__ import annotations

import torch


def critic_salience_weight(contact_probability: torch.Tensor,
                           delta_world: torch.Tensor, *, coefficient: float,
                           effect_scale_m: float = .003) -> torch.Tensor:
    """Return bounded state-level weights; do not alter actor advantages."""
    if (contact_probability.ndim != 1 or
            delta_world.shape != (contact_probability.numel(), 3) or
            not 0 < coefficient <= 2 or effect_scale_m <= 0):
        raise ValueError("invalid Cm critic salience input")
    if (not torch.isfinite(contact_probability).all() or
            not torch.isfinite(delta_world).all() or
            not ((contact_probability >= 0) & (contact_probability <= 1)).all()):
        raise FloatingPointError("invalid Cm critic prediction")
    effect = (delta_world[:, 2].abs() / effect_scale_m).clamp(0, 1)
    return (1 + coefficient * contact_probability * effect).detach()


def normalized_critic_loss(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Preserve the minibatch's mean critic scale while changing sample focus."""
    if loss.ndim not in (1, 2) or weight.ndim != 1 or loss.shape[0] != weight.numel():
        raise ValueError("critic loss and salience weight batch mismatch")
    if not torch.isfinite(weight).all() or not (weight >= 1).all():
        raise FloatingPointError("invalid Cm critic salience weight")
    normalized = weight / weight.mean()
    if loss.ndim == 2:
        normalized = normalized[:, None]
    return loss * normalized
