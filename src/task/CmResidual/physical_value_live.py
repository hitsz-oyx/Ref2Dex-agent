"""Live simulator adapter and common reward for the physical-value route."""
from __future__ import annotations
import torch
from src.task.CmResidual.physical_value_contract import canonical_quaternion
from src.task.CmResidual.dexplore_grasp_reward import held_lift_reward, contact_lift_progress_reward
from src.task.CmResidual.dexplore_approach import potential_approach_reward


def contacts(task):
    hand = (task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(-1)
    obj = task._tar_contact_forces.norm(dim=-1) > .1
    return torch.stack((hand, obj), -1).float()


def snapshot(task, tracker):
    obj = task._target_states.clone()
    obj[:, 3:7] = canonical_quaternion(obj[:, 3:7])
    return torch.cat((task._dof_pos.clone(), task._dof_vel.clone(), obj,
                      contacts(task), tracker.events.clone()), -1)


def context(task, tracker, delta=0):
    phase = task.progress_buf.clone() + delta
    length = task.max_episode_length[task.data_id].clone()
    safe = torch.minimum(phase, length - 1)
    ref = task.hoi_data[task.data_id, safe].clone()
    # Reference is explicit, not a fabricated future actor observation.
    meta = torch.cat((torch.nn.functional.one_hot(task.data_id, 3).float(),
                      (phase / length)[:, None],
                      ((length - phase) / length)[:, None],
                      (task.start_times / length)[:, None],
                      tracker.initial_height[:, None]), -1)
    return torch.cat((meta, ref), -1)


def shared_reward(task, tracker, original_reward, previous_z, gap_before, gap_after, gamma, approach_config):
    """Match the source's existing 2/10/5 shaping, plus approved hold/drop."""
    pair = contacts(task).bool().all(-1)
    done = task.reset_buf.bool()
    base = original_reward.reshape(-1)
    approach = 2 * potential_approach_reward(gap_before, gap_after, done,
                                             gamma=gamma, config=approach_config)
    rest_z = task.hoi_refs[task.data_id, task.ref_index, 0, 108]
    held = 10 * held_lift_reward(task._target_states[:, 2], rest_z, pair, torch.ones_like(pair))
    progress = 5 * contact_lift_progress_reward(previous_z, task._target_states[:, 2],
                                                pair, torch.ones_like(pair), done)
    stable, _ = tracker.step(task._target_states[:, 2], pair)
    components = torch.stack((base, approach, held, progress, stable), -1)
    return components.sum(-1), components
