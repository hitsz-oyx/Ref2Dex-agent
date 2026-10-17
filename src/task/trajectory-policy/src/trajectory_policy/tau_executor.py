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
REFERENCE_ANCHOR_DIM = 33
ACTION_DIM = 12
TASK_GOAL_DIM = 12
CONTACT_CONTEXT_DIM = 10
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


DIRECT_SCHEMA = "ref2dex.direct-tau-executor.v1"


def validate_direct_batch(state: torch.Tensor, tau: torch.Tensor,
                          goal: torch.Tensor,
                          previous_action: torch.Tensor) -> None:
    """Validate the direct executor contract, including the task goal."""
    validate_batch(state, tau, previous_action)
    batch = state.shape[0]
    _require("goal", goal, (batch, TASK_GOAL_DIM))
    if goal.device != state.device:
        raise ValueError("state, tau, goal and previous_action must share a device")


class DirectTauExecutorPolicy(nn.Module):
    """Closed-loop policy mapping measured state and a future GT-τ plan to action.

    The goal is deliberately a compact 12-D object pose (the task target pose
    flattened as the first three rows of its homogeneous transform).  The
    policy emits only active Inspire coordinates; passive coupling is owned by
    the simulator adapter.
    """

    schema = DIRECT_SCHEMA

    def __init__(self, horizon: int = 8, width: int = 256):
        super().__init__()
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(width, int) or width < 16:
            raise ValueError("width must be an integer >= 16")
        self.horizon = horizon
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM + TASK_GOAL_DIM, width), nn.Tanh(),
        )
        self.tau_encoder = nn.Sequential(
            nn.Linear(HAND_PLAN_DIM, width), nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * width, width), nn.Tanh(),
            nn.Linear(width, ACTION_DIM),
        )

    def forward(self, state: torch.Tensor, tau: torch.Tensor,
                goal: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
        validate_direct_batch(state, tau, goal, previous_action)
        if tau.shape[1] != self.horizon:
            raise ValueError(
                f"tau horizon mismatch: model={self.horizon}, batch={tau.shape[1]}"
            )
        state_code = self.state_encoder(
            torch.cat((state, goal, previous_action), dim=-1)
        )
        tau_code = self.tau_encoder(tau).mean(dim=1)
        output = torch.tanh(self.head(torch.cat((state_code, tau_code), dim=-1)))
        if not torch.isfinite(output).all():
            raise FloatingPointError("nonfinite direct executor action")
        return output


ORACLE_C_SCHEMA = "ref2dex.direct-tau-executor-oracle-c.v1"


class DirectTauExecutorOracleCPolicy(DirectTauExecutorPolicy):
    """Direct executor augmented with the measured interaction context C.

    C is intentionally an explicit oracle branch for the next diagnostic, not
    part of the GT-τ baseline: pair/contact gaps, support gap, object velocity,
    and table-footprint indicators measured at the current query state.
    """

    schema = ORACLE_C_SCHEMA

    def __init__(self, horizon: int = 8, width: int = 256):
        super(DirectTauExecutorPolicy, self).__init__()
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(width, int) or width < 16:
            raise ValueError("width must be an integer >= 16")
        self.horizon = horizon
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM + TASK_GOAL_DIM + CONTACT_CONTEXT_DIM, width), nn.Tanh(),
        )
        self.tau_encoder = nn.Sequential(nn.Linear(HAND_PLAN_DIM, width), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, ACTION_DIM))

    def forward(self, state, tau, goal, previous_action, contact):
        validate_direct_batch(state, tau, goal, previous_action)
        batch = state.shape[0]
        _require("contact", contact, (batch, CONTACT_CONTEXT_DIM))
        if contact.device != state.device:
            raise ValueError("contact must share the policy device")
        if tau.shape[1] != self.horizon:
            raise ValueError(f"tau horizon mismatch: model={self.horizon}, batch={tau.shape[1]}")
        state_code = self.state_encoder(torch.cat((state, goal, previous_action, contact), dim=-1))
        tau_code = self.tau_encoder(tau).mean(dim=1)
        output = torch.tanh(self.head(torch.cat((state_code, tau_code), dim=-1)))
        if not torch.isfinite(output).all():
            raise FloatingPointError("nonfinite oracle-C executor action")
        return output


TEMPORAL_SCHEMA = "ref2dex.direct-tau-executor.temporal-v2"


class TemporalTauExecutorPolicy(nn.Module):
    """Direct executor with an order-sensitive reference-plan encoder.

    This r3 contract intentionally has no task-goal input.  The current probe
    asks whether a measured state can execute a reference plan; goal
    conditioning is deferred until there are multiple goal-bearing sequences.
    """

    schema = TEMPORAL_SCHEMA

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
        self.tau_input = nn.Sequential(nn.Linear(HAND_PLAN_DIM, width), nn.Tanh())
        self.position = nn.Parameter(torch.zeros(1, horizon, width))
        nn.init.normal_(self.position, mean=0., std=.02)
        # Flatten after adding a learned position code.  This is deliberately
        # a temporal MLP rather than GRU: the archived DExplore runtime has an
        # incompatible cuDNN shared library, while this encoder remains fully
        # GPU-compatible and order-sensitive.
        self.tau_encoder = nn.Sequential(
            nn.Linear(horizon * width, width), nn.Tanh(),
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
        encoded = self.tau_input(tau) + self.position
        tau_code = self.tau_encoder(encoded.reshape(len(encoded), -1))
        output = torch.tanh(self.head(torch.cat((state_code, tau_code), dim=-1)))
        if not torch.isfinite(output).all():
            raise FloatingPointError("nonfinite temporal executor action")
        return output


ABSOLUTE_PLAN_SCHEMA = "ref2dex.direct-tau-executor.absolute-plan-v3"


class AbsolutePlanTauExecutorPolicy(nn.Module):
    """Direct executor with an absolute reference-hand anchor.

    ``anchor`` is the reference hand at the current tick expressed in the
    reference object frame.  ``tau`` remains the ordered future displacement
    plan.  The two inputs distinguish where the reference currently is from
    how it will move next, which is required once the learner leaves the
    teacher manifold.
    """

    schema = ABSOLUTE_PLAN_SCHEMA

    def __init__(self, horizon: int = 8, width: int = 256):
        super().__init__()
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(width, int) or width < 16:
            raise ValueError("width must be an integer >= 16")
        self.horizon = horizon
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM + REFERENCE_ANCHOR_DIM, width), nn.Tanh(),
        )
        self.tau_input = nn.Sequential(nn.Linear(HAND_PLAN_DIM, width), nn.Tanh())
        self.position = nn.Parameter(torch.zeros(1, horizon, width))
        nn.init.normal_(self.position, mean=0., std=.02)
        self.tau_encoder = nn.Sequential(nn.Linear(horizon * width, width), nn.Tanh())
        self.head = nn.Sequential(
            nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, ACTION_DIM),
        )

    def forward(self, state: torch.Tensor, tau: torch.Tensor,
                anchor: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or tau.ndim != 3 or anchor.ndim != 2 or previous_action.ndim != 2:
            raise ValueError("state, tau, anchor and previous_action must be batched")
        batch, horizon = state.shape[0], tau.shape[1]
        _require("state", state, (batch, STATE_DIM))
        _require("tau", tau, (batch, horizon, HAND_PLAN_DIM))
        _require("anchor", anchor, (batch, REFERENCE_ANCHOR_DIM))
        _require("previous_action", previous_action, (batch, ACTION_DIM))
        if horizon != self.horizon:
            raise ValueError(f"tau horizon mismatch: model={self.horizon}, batch={horizon}")
        if (state.device != tau.device or state.device != anchor.device
                or state.device != previous_action.device):
            raise ValueError("all executor inputs must share a device")
        state_code = self.state_encoder(torch.cat((state, anchor, previous_action), dim=-1))
        encoded = self.tau_input(tau) + self.position
        tau_code = self.tau_encoder(encoded.reshape(batch, -1))
        output = torch.tanh(self.head(torch.cat((state_code, tau_code), dim=-1)))
        if not torch.isfinite(output).all():
            raise FloatingPointError("nonfinite absolute-plan executor action")
        return output
