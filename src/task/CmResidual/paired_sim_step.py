"""Validated one-step physical action pair from an Isaac Gym task snapshot.

No Isaac Gym import here so the snapshot/restore contract can be CPU-tested.
"""
from __future__ import annotations

from collections.abc import Callable

import torch


def paired_sim_step(task, base_action: torch.Tensor, alternate_action: torch.Tensor,
                    unwrap_tensor: Callable[[torch.Tensor], object],
                    *, restore_tolerance: float = 1e-6,
                    repeat_object_tolerance_m: float = 5e-5,
                    repeat_joint_tolerance: float = 1e-4) -> dict[str, torch.Tensor]:
    """Run base twice and alternate once from exactly one visible physics state.

    The routine is invalid if the simulator cannot restore actor roots/DOFs or
    replay the same action within tolerance.  Contact-solver hidden state is
    deliberately tested by the repeated base action, not assumed restored.
    """
    if (base_action.ndim != 2 or base_action.shape != alternate_action.shape or
            base_action.shape != task._dof_pos.shape):
        raise ValueError("actions must match native simulator DOF positions [B,18]")
    if not torch.isfinite(base_action).all() or not torch.isfinite(alternate_action).all():
        raise FloatingPointError("paired actions must be finite")
    if not ((alternate_action - base_action).abs().sum(dim=-1) > 1e-6).any():
        raise ValueError("paired candidate actions do not differ")
    root_before = task._root_states.clone()
    dof_before = task._dof_state.clone()
    rigid_before = task._rigid_body_state.clone()
    q_before = task._dof_pos.clone()
    object_before = task._target_states.clone()

    def restore() -> tuple[torch.Tensor, torch.Tensor]:
        task._root_states.copy_(root_before)
        task._dof_state.copy_(dof_before)
        task.gym.set_actor_root_state_tensor(task.sim, unwrap_tensor(task._root_states))
        task.gym.set_dof_state_tensor(task.sim, unwrap_tensor(task._dof_state))
        task._refresh_sim_tensors()
        root_error = (task._root_states - root_before).abs().amax()
        dof_error = (task._dof_state - dof_before).abs().amax()
        rigid_error = (task._rigid_body_state - rigid_before).abs().amax()
        if (not torch.isfinite(root_error + dof_error + rigid_error) or
                root_error > restore_tolerance or dof_error > restore_tolerance or
                rigid_error > restore_tolerance):
            raise RuntimeError(
                "physics state restore failed: "
                f"root={float(root_error)}, dof={float(dof_error)}, "
                f"rigid={float(rigid_error)}")
        return root_error, dof_error, rigid_error

    def advance(action):
        task.pre_physics_step(action)
        task._physics_step()
        task._refresh_sim_tensors()
        return task._dof_pos.clone(), task._target_states.clone()

    try:
        base_q, base_object = advance(base_action)
        restore()
        repeat_q, repeat_object = advance(base_action)
        restore()
        alt_q, alt_object = advance(alternate_action)
    finally:
        restore()

    repeated_object_mm = (repeat_object[:, :3] - base_object[:, :3]).norm(dim=-1) * 1000
    repeated_joint = (repeat_q - base_q).abs().amax(dim=-1)
    if ((repeated_object_mm > repeat_object_tolerance_m * 1000).any() or
            (repeated_joint > repeat_joint_tolerance).any()):
        raise RuntimeError(
            "same-action physics replay is not deterministic enough: "
            f"object_p95_mm={float(torch.quantile(repeated_object_mm, .95)):.6f}, "
            f"object_max_mm={float(repeated_object_mm.max()):.6f}, "
            f"joint_max={float(repeated_joint.max()):.6f}")
    result = {
        "q": q_before,
        "object_state": object_before,
        "base_action": base_action.clone(),
        "alternate_action": alternate_action.clone(),
        "base_next_q": base_q,
        "alternate_next_q": alt_q,
        "base_next_object_state": base_object,
        "alternate_next_object_state": alt_object,
        "same_action_repeat_object_mm": repeated_object_mm,
        "same_action_repeat_joint_max": repeated_joint,
        "actual_effect_mm": (alt_object[:, :3] - base_object[:, :3]).norm(dim=-1) * 1000,
        "action_gap_l2": (alternate_action - base_action).norm(dim=-1),
    }
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("non-finite paired physical transition")
    return result
