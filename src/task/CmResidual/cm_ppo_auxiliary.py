"""Frozen-Cm supervision for PPO actor features; no reward/action override."""
from __future__ import annotations

import torch

from src.task.CmResidual.cmlite import local_to_world_translation


def raw_cm_input(q, dof_vel, object_state, action):
    if (q.ndim != 2 or q.shape[1] != 18 or action.shape != q.shape or
            dof_vel.shape != q.shape or object_state.shape != (len(q), 13)):
        raise ValueError("expected q/velocity/action [B,18], object [B,13]")
    relative_q = q.clone()
    relative_q[:, :3] -= object_state[:, :3]
    return torch.cat((relative_q, action, dof_vel, object_state), dim=-1)


@torch.no_grad()
def cm_candidate_targets(cm, q, dof_vel, object_state, action,
                         contact_mask, delta=.1):
    """Three normalized pre-treatment contrasts from a frozen contact Cm."""
    if contact_mask.shape != (len(q),) or contact_mask.dtype != torch.bool:
        raise ValueError("boolean contact mask [B] required")
    if not 0 < delta <= .5:
        raise ValueError("invalid candidate dose")
    candidate = action.detach().clamp(-1, 1)
    plus, minus = candidate.clone(), candidate.clone()
    plus[:, 2] = (plus[:, 2] + delta).clamp(-1, 1)
    minus[:, 2] = (minus[:, 2] - delta).clamp(-1, 1)
    raw = raw_cm_input(torch.cat((q, q)), torch.cat((dof_vel, dof_vel)),
                       torch.cat((object_state, object_state)),
                       torch.cat((plus, minus)))
    predicted = cm(raw)
    count = len(q)
    contact = (predicted["contact_fraction"][count:] -
               predicted["contact_fraction"][:count]) / .1
    step_local = (predicted["delta_local"][:count] -
                  predicted["delta_local"][count:])
    future_local = (predicted["followup_delta_local"][:count] -
                    predicted["followup_delta_local"][count:])
    step = local_to_world_translation(object_state, step_local)[:, 2] / .02
    future = local_to_world_translation(object_state, future_local)[:, 2] / .02
    target = torch.stack((contact, step, future), dim=-1).clamp(-2, 2)
    target[~contact_mask] = 0
    if not torch.isfinite(target).all():
        raise FloatingPointError("non-finite Cm auxiliary target")
    return target


@torch.no_grad()
def cm_h10_multiaxis_targets(cm, q, dof_vel, object_state, action,
                            contact_mask, delta=.1):
    """Frozen H10 Cm central contrasts for wrist xyz: world xyz and contact.

    Target order is (object dx, dy, dz, contact) for each wrist axis.
    Only unclipped ±delta contrasts at existing contact are supervised.
    """
    if contact_mask.shape != (len(q),) or contact_mask.dtype != torch.bool:
        raise ValueError("boolean contact mask [B] required")
    if not 0 < delta <= .5:
        raise ValueError("invalid candidate dose")
    candidate = action.detach().clamp(-1, 1)
    safe = (candidate[:, :3].abs() <= 1 - delta).all(dim=-1)
    variants = []
    for axis in range(3):
        for sign in (1, -1):
            variant = candidate.clone()
            variant[:, axis] = (variant[:, axis] + sign * delta).clamp(-1, 1)
            variants.append(variant)
    raw = raw_cm_input(q.repeat(6, 1), dof_vel.repeat(6, 1),
                       object_state.repeat(6, 1), torch.cat(variants))
    predicted = cm(raw)
    if predicted["followup_delta_local"].shape != (6 * len(q), 3) or (
            predicted["contact_fraction"].shape != (6 * len(q),)):
        raise ValueError("H10 Cm prediction shape mismatch")
    followup = local_to_world_translation(
        object_state.repeat(6, 1), predicted["followup_delta_local"])
    followup = followup.reshape(6, len(q), 3)
    contact = predicted["contact_fraction"].reshape(6, len(q))
    blocks = []
    for axis in range(3):
        translation = (followup[2 * axis] - followup[2 * axis + 1]) / .02
        contact_effect = ((contact[2 * axis] - contact[2 * axis + 1]) / .1)[:, None]
        blocks.append(torch.cat((translation, contact_effect), dim=-1))
    target = torch.cat(blocks, dim=-1).clamp(-2, 2)
    target[~(contact_mask & safe)] = 0
    if not torch.isfinite(target).all():
        raise FloatingPointError("non-finite H10 Cm auxiliary target")
    return target, contact_mask & safe


def masked_auxiliary_loss(prediction, target, contact_mask):
    """Per-row loss whose batch mean equals mean contact-row MSE."""
    if prediction.ndim != 2 or prediction.shape[1] < 1 or (
            target.shape != prediction.shape or
            contact_mask.shape != (len(prediction), 1)):
        raise ValueError("auxiliary prediction/target/mask shape mismatch")
    if not all(torch.isfinite(value).all() for value in (prediction, target, contact_mask)):
        raise FloatingPointError("non-finite Cm auxiliary loss input")
    if (contact_mask < 0).any() or (contact_mask > 1).any():
        raise ValueError("invalid auxiliary mask")
    per_row = (prediction - target).square().mean(dim=-1, keepdim=True)
    return per_row * contact_mask / contact_mask.mean().clamp_min(.1)
