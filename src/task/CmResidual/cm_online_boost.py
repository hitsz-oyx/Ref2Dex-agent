"""Bounded contact-gated action modification for online Cm probes."""
from __future__ import annotations

import torch


def apply_wrist_z_boost(action: torch.Tensor, selected: torch.Tensor,
                        delta_z: float = .1) -> torch.Tensor:
    if action.ndim != 2 or action.shape[1] != 18 or selected.shape != (len(action),):
        raise ValueError("action [B,18] and selected [B] required")
    if selected.dtype != torch.bool or selected.device != action.device:
        raise ValueError("selected must be boolean on the action device")
    if not 0 < delta_z <= .5 or not torch.isfinite(action).all():
        raise ValueError("invalid delta or non-finite actions")
    output = action.clone()
    output[selected, 2] = (output[selected, 2] + delta_z).clamp(-1, 1)
    return output
