"""Measured-state, future-hand-plan executor for the GT-τ qualification probe.

This module is deliberately independent of the old nominal/residual redirector.
It defines the smallest BC/PPO-ready contract: the policy reads the measured
robot state, a future hand plan expressed in the live object frame, and the
previous command that was actually applied.  It emits only the twelve native
active coordinates.  The Inspire adapter remains the single owner of passive
coordinate coupling and final clipping.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


SCHEMA = "ref2dex.measured-state-tau-executor.v1"
STATE_DIM = 87
HAND_PLAN_DIM = 33
ACTION_DIM = 12
ACTIVE_ACTION = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15)


def _require(name: str, value: torch.Tensor, shape: Tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating point tensor")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def validate_batch(state: torch.Tensor, tau: torch.Tensor,
                   previous_action: torch.Tensor) -> None:
    """Validate the public policy contract for one batched query."""
    if state.ndim != 2 or tau.ndim != 3 or previous_action.ndim != 2:
        raise ValueError("state, tau and previous_action must be batched tensors")
    batch, horizon = state.shape[0], tau.shape[1]
    _require("state", state, (batch, STATE_DIM))
    _require("tau", tau, (batch, horizon, HAND_PLAN_DIM))
    _require("previous_action", previous_action, (batch, ACTION_DIM))
    if horizon < 1:
        raise ValueError("tau must contain at least one future step")
    if state.device != tau.device or state.device != previous_action.device:
        raise ValueError("state, tau and previous_action must share a device")


def expand_active_action(active: torch.Tensor) -> torch.Tensor:
    """Place twelve active coordinates in the 18-D native action layout.

    Passive coordinates are intentionally zero.  The robot adapter applies its
    fixed coupling map exactly once after this function; duplicating that map
    in a learned policy would recreate the old redirector's double-application
    failure mode.
    """
    if not isinstance(active, torch.Tensor) or active.ndim != 2:
        raise ValueError("active action must be a [batch, 12] tensor")
    if active.shape[-1] != ACTION_DIM or not torch.isfinite(active).all():
        raise ValueError("active action must be finite with twelve coordinates")
    result = active.new_zeros((active.shape[0], 18))
    result[:, list(ACTIVE_ACTION)] = active
    return result


class MeasuredStateTauPolicy(nn.Module):
    """Compact policy suitable for BC initialization and later PPO."""

    schema = SCHEMA

    def __init__(self, horizon: int = 8, width: int = 256):
        super().__init__()
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(width, int) or width < 16:
            raise ValueError("width must be an integer >= 16")
        self.horizon = horizon
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, width), nn.Tanh(),
        )
        self.tau_encoder = nn.Sequential(
            nn.Linear(HAND_PLAN_DIM, width), nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * width, width), nn.Tanh(),
            nn.Linear(width, ACTION_DIM),
        )

    def forward(self, state: torch.Tensor, tau: torch.Tensor,
                previous_action: torch.Tensor) -> torch.Tensor:
        validate_batch(state, tau, previous_action)
        if tau.shape[1] != self.horizon:
            raise ValueError(
                f"tau horizon mismatch: model={self.horizon}, batch={tau.shape[1]}"
            )
        state_code = self.state_encoder(torch.cat((state, previous_action), dim=-1))
        # Mean pooling preserves a fixed model size while retaining all plan
        # steps.  The temporal encoder can be replaced later without changing
        # the data or action contract.
        tau_code = self.tau_encoder(tau).mean(dim=1)
        output = torch.tanh(self.head(torch.cat((state_code, tau_code), dim=-1)))
        if not torch.isfinite(output).all():
            raise FloatingPointError("nonfinite executor action")
        return output

