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


def permute_active_weights(weights: torch.Tensor, actor_mask: torch.Tensor,
                           generator: torch.Generator) -> torch.Tensor:
    """Placebo: retain each step's active-weight multiset, break sample alignment."""
    if weights.ndim != 1 or actor_mask.shape != weights.shape:
        raise ValueError("Cm weights and PPO actor mask must be matching vectors")
    if not torch.isfinite(weights).all() or not (weights >= 1).all():
        raise FloatingPointError("Cm PPO weights must be finite and at least one")
    if not ((actor_mask == 0) | (actor_mask == 1)).all():
        raise ValueError("PPO actor mask must be binary")
    active = actor_mask.bool()
    result = weights.clone()
    selected = weights[active]
    if selected.numel() > 1:
        order = torch.randperm(selected.numel(), generator=generator,
                               device=selected.device)
        result[active] = selected[order]
    return result.detach()


def align_active_weight_multiset(weights: torch.Tensor, score: torch.Tensor,
                                 actor_mask: torch.Tensor) -> torch.Tensor:
    """Keep joint weights' active multiset, assign it by one component's rank.

    This isolates which model head selects PPO samples while holding the
    per-step active weight distribution exactly fixed.
    """
    if weights.ndim != 1 or score.shape != weights.shape or actor_mask.shape != weights.shape:
        raise ValueError("weights, score and actor mask must be matching vectors")
    if not (torch.isfinite(weights).all() and torch.isfinite(score).all()):
        raise FloatingPointError("weights and score must be finite")
    if not ((actor_mask == 0) | (actor_mask == 1)).all():
        raise ValueError("PPO actor mask must be binary")
    active = actor_mask.bool()
    result = weights.clone()
    if int(active.sum()) > 1:
        order = torch.argsort(score[active], stable=True)
        sorted_weights = torch.sort(weights[active]).values
        assigned = torch.empty_like(sorted_weights)
        assigned[order] = sorted_weights
        result[active] = assigned
    return result.detach()
