"""Frozen ten-step Cm rule for choosing a single contact-window action."""
from __future__ import annotations

import torch


def select_x_or_z(world_future: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
    """Return 0=base, 1=wrist x+, 2=wrist z+ for [base,x+,z+] rows."""
    if (world_future.ndim != 3 or world_future.shape[0] != 3 or
            world_future.shape[2] != 3 or contact.shape != world_future.shape[:2]):
        raise ValueError("expected [3,B,3] future motion and [3,B] contact")
    if not torch.isfinite(world_future).all() or not torch.isfinite(contact).all():
        raise FloatingPointError("non-finite Cm candidate predictions")
    if (contact < 0).any() or (contact > 1).any():
        raise ValueError("contact prediction outside [0,1]")
    z_gain = world_future[2, :, 2] - world_future[0, :, 2]
    z_contact = contact[2] - contact[0]
    x_gain = world_future[1, :, 2] - world_future[0, :, 2]
    x_contact = contact[1] - contact[0]
    choose_z = (z_gain >= .005) & (z_contact >= -.02)
    choose_x = ~choose_z & (x_contact >= .03) & (x_gain >= -.005)
    result = torch.zeros(world_future.shape[1], dtype=torch.int8,
                         device=world_future.device)
    result[choose_x] = 1
    result[choose_z] = 2
    return result


def apply_choice(action: torch.Tensor, choice: torch.Tensor, delta=.1) -> torch.Tensor:
    if (action.ndim != 2 or action.shape[1] != 18 or
            choice.shape != (len(action),) or
            not 0 < delta <= .5 or
            not torch.isin(choice, torch.tensor([0, 1, 2], device=choice.device)).all()):
        raise ValueError("invalid multiaxis action choice")
    modified = action.detach().clone()
    modified[choice == 1, 0] += delta
    modified[choice == 2, 2] += delta
    if ((modified[choice == 1, 0] > 1 + 1e-6).any() or
            (modified[choice == 2, 2] > 1 + 1e-6).any()):
        raise ValueError("chosen action clips")
    return modified
