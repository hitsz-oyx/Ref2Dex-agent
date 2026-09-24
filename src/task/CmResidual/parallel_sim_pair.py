"""One-step physical action comparison in three matched simulator environments.

Unlike sequential snapshot/restore, each arm advances once from its own
independently evolved physics state. A same-action third arm bounds hidden
solver/environment differences before action-effect data are accepted.
"""
from __future__ import annotations

from collections.abc import Callable

import torch


def parallel_sim_pair_step(task, action: torch.Tensor, env_step: Callable,
                           *, delta_z: float = .1,
                           prestate_tolerance: float = 1e-4,
                           repeat_object_tolerance_mm: float = .05,
                           repeat_joint_tolerance: float = 1e-4):
    """Run base/repeat/alternate arms once; reject unmatched physics states."""
    batch = action.shape[0]
    if batch < 3 or batch % 3 or action.ndim != 2 or action.shape[1] != task.num_dof:
        raise ValueError("parallel pair requires triplets of native DOF actions")
    if not 0 < delta_z <= .5 or not torch.isfinite(action).all():
        raise ValueError("invalid action perturbation or non-finite actions")
    base = torch.arange(0, batch, 3, device=action.device)
    repeat, alternate = base + 1, base + 2
    root = task._root_states.view(batch, -1, 13)
    dof = task._dof_state.view(batch, -1, 2)
    rigid = task._rigid_body_state.view(batch, -1, 13)
    prestate_gap = {}
    for name, tensor in (("root", root), ("dof", dof), ("rigid", rigid)):
        prestate_gap[name] = torch.maximum(
            (tensor[repeat] - tensor[base]).abs().flatten(1).amax(1),
            (tensor[alternate] - tensor[base]).abs().flatten(1).amax(1))
    for name in ("progress_buf", "data_id", "start_times", "ref_index"):
        value = getattr(task, name)
        prestate_gap[name] = torch.maximum(
            (value[repeat] - value[base]).abs().reshape(len(base), -1).amax(1),
            (value[alternate] - value[base]).abs().reshape(len(base), -1).amax(1))
    worst = max(float(value.max()) for value in prestate_gap.values())
    if not torch.isfinite(torch.tensor(worst)) or worst > prestate_tolerance:
        details = {name: float(value.max()) for name, value in prestate_gap.items()}
        raise RuntimeError(f"parallel physics prestates differ: {details}")
    before_q = task._dof_pos[base].clone()
    before_object = task._target_states[base].clone()
    before_object_repeat = task._target_states[repeat].clone()
    before_object_alternate = task._target_states[alternate].clone()
    pre_contact = ((task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(-1) &
                   (task._tar_contact_forces.norm(dim=-1) > .1))[base].clone()
    progress = task.progress_buf[base].clone()
    motion_id = task.data_id[base].clone()
    executed = action.clone()
    executed[repeat] = executed[base]
    executed[alternate] = executed[base]
    executed[alternate, 2] = (executed[alternate, 2] + delta_z).clamp(-1, 1)
    action_gap = (executed[alternate] - executed[base]).norm(dim=-1)
    if (action_gap <= .001).any():
        raise RuntimeError("alternate action is clamped to base action")
    env_output = env_step(executed)
    base_q = task._dof_pos[base].clone()
    repeat_q = task._dof_pos[repeat].clone()
    alternate_q = task._dof_pos[alternate].clone()
    base_object = task._target_states[base].clone()
    repeat_object = task._target_states[repeat].clone()
    alternate_object = task._target_states[alternate].clone()
    base_delta = base_object[:, :3] - before_object[:, :3]
    repeat_delta = repeat_object[:, :3] - before_object_repeat[:, :3]
    alternate_delta = alternate_object[:, :3] - before_object_alternate[:, :3]
    repeat_object_mm = (repeat_delta - base_delta).norm(dim=-1) * 1000
    repeat_joint = (repeat_q - base_q).abs().amax(dim=-1)
    if (repeat_object_mm > repeat_object_tolerance_mm).any() or (
            repeat_joint > repeat_joint_tolerance).any():
        raise RuntimeError(
            "same-action parallel replay differs: "
            f"object_max_mm={float(repeat_object_mm.max()):.6f}, "
            f"joint_max={float(repeat_joint.max()):.6f}")
    record = {
        "q": before_q, "object_state": before_object,
        "base_action": executed[base].clone(),
        "alternate_action": executed[alternate].clone(),
        "base_next_q": base_q, "alternate_next_q": alternate_q,
        "base_next_object_state": base_object,
        "alternate_next_object_state": alternate_object,
        "pre_contact": pre_contact, "progress": progress,
        "motion_id": motion_id,
        "same_action_repeat_object_mm": repeat_object_mm,
        "same_action_repeat_joint_max": repeat_joint,
        "actual_effect_mm": (alternate_delta - base_delta).norm(dim=-1) * 1000,
        "effect_dz_mm": (alternate_delta[:, 2] - base_delta[:, 2]) * 1000,
        "action_gap_l2": action_gap,
        "pre_root_gap": prestate_gap["root"],
        "pre_dof_gap": prestate_gap["dof"],
        "pre_rigid_gap": prestate_gap["rigid"],
    }
    if not all(torch.isfinite(value).all() for value in record.values()):
        raise FloatingPointError("non-finite parallel pair")
    return env_output, record
