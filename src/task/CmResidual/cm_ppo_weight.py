"""Bounded action-conditioned world-model weights for PPO actor updates."""
from __future__ import annotations

import torch


def effect_actor_weight(contact_probability: torch.Tensor,
                        delta_world: torch.Tensor, *, coefficient: float,
                        effect_scale_m: float = 0.003) -> torch.Tensor:
    """Emphasize potentially consequential actions without changing reward sign.

    The detached, bounded weight multiplies PPO's actor surrogate for the
    *recorded* action. Real advantages still decide whether that action should
    become more or less likely. The world model cannot supply a fake reward.
    """
    if (contact_probability.ndim != 1 or delta_world.shape !=
            (contact_probability.shape[0], 3) or coefficient < 0 or
            effect_scale_m <= 0):
        raise ValueError("invalid Cm PPO weight input")
    if not (torch.isfinite(contact_probability).all() and
            torch.isfinite(delta_world).all() and
            bool(((contact_probability >= 0) & (contact_probability <= 1)).all())):
        raise FloatingPointError("non-finite or out-of-range Cm prediction")
    effect = (delta_world[:, 2].abs() / effect_scale_m).clamp(0, 1)
    return (1.0 + coefficient * contact_probability * effect).detach()
