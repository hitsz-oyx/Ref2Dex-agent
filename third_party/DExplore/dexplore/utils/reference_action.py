"""Reference joint-space action diagnostic for the Inspire task."""
from __future__ import annotations

import torch


def inspire_reference_action(task, lead: int = 1) -> torch.Tensor:
    """Map a future reference pose to DExplore's normalized PD action."""
    if lead < 0:
        raise ValueError("reference action lead must be nonnegative")
    final_frame = task.max_episode_length[task.data_id] - 1
    frame = torch.minimum(task.progress_buf + lead, final_frame)
    reference = task.hoi_refs[
        task.data_id, task.ref_index, frame, 119:119 + task.num_dof]
    action = torch.empty_like(reference)
    action[:, :6] = ((reference[:, :6] - task._dof_pos[:, :6]) /
                     task._pd_action_scale[:6])
    absolute = ((reference[:, 6:] - task._pd_action_offset[6:]) /
                task._pd_action_scale[6:])
    action[:, 6:] = 2.0 * absolute - 1.0
    return action.clamp(-1.0, 1.0)
