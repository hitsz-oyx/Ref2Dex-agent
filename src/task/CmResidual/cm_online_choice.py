"""Pre-fixed contact-aware Cm gate for a bounded wrist-z action candidate."""
from __future__ import annotations

import torch

from src.task.CmResidual.cmlite import local_to_world_translation


def select_contact_preserving_down(base_contact, down_contact,
                                   base_followup_local, down_followup_local,
                                   object_state, *, min_contact_gain=.05,
                                   max_lift_loss_m=.01):
    if (base_contact.shape != down_contact.shape or
            base_contact.ndim != 1 or
            base_followup_local.shape != down_followup_local.shape or
            base_followup_local.shape != (len(base_contact), 3) or
            object_state.shape != (len(base_contact), 13)):
        raise ValueError("Cm candidate prediction shape mismatch")
    if not all(torch.isfinite(value).all() for value in (
            base_contact, down_contact, base_followup_local,
            down_followup_local, object_state)):
        raise FloatingPointError("non-finite Cm candidate prediction")
    if not (0 <= min_contact_gain <= 1 and 0 <= max_lift_loss_m <= .1):
        raise ValueError("invalid Cm action gate")
    contact_gain = down_contact - base_contact
    local_delta = down_followup_local - base_followup_local
    lift_change = local_to_world_translation(object_state, local_delta)[:, 2]
    return (contact_gain >= min_contact_gain) & (lift_change >= -max_lift_loss_m)


def apply_wrist_z_down(action, selected, delta_z=.1):
    if action.ndim != 2 or action.shape[1] != 18 or selected.shape != (len(action),):
        raise ValueError("action [B,18] and selected [B] required")
    if selected.dtype != torch.bool or selected.device != action.device:
        raise ValueError("selected mask must be boolean on action device")
    if not 0 < delta_z <= .5 or not torch.isfinite(action).all():
        raise ValueError("invalid action or dose")
    result = action.clone()
    result[selected, 2] = (result[selected, 2] - delta_z).clamp(-1, 1)
    return result
