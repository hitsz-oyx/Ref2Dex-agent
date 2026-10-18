"""Canonical-action isolation for the pinned UniDexGrasp runtime.

UniDexGrasp's ``pre_physics_step`` rewrites the first six wrist action
channels in place from the current yaw frame into the world frame.  A caller
that records the tensor after the simulator call therefore stores a different
action than the one it issued, and replaying that prefix rotates the wrist
twice.  Keep the canonical command outside the simulator's ownership.
"""
from __future__ import annotations


def canonical_action(action):
    """Detach a copy before the simulator can mutate its argument."""
    return action.detach().clone()


def step_with_isolated_action(sim_step, action):
    """Run one simulator step without giving it ownership of ``action``."""
    sim_step(action.clone())
