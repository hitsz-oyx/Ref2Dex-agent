"""Explicit current-state and planned-reference actor context."""
import torch
from src.task.CmResidual.physical_value_live import contacts
NULL_COMMANDS=(7,9,11,13,16,17)
SCHEMA='hold70-v1'
ACTION_SCALES=(.005,)*3+(.05,)*3+(.1,)*12


def hold_context(task,phase_stop):
    motion=task.data_id;next_progress=torch.minimum(task.progress_buf+1,phase_stop[motion])
    planned=task.hoi_refs[motion,task.ref_index,next_progress,119:137]
    phase=(next_progress.float()/phase_stop[motion])[:,None]
    context=torch.cat((task._dof_pos,task._dof_vel,task._target_states,contacts(task),planned,phase),-1)
    if context.shape!=(task.num_envs,70) or not torch.isfinite(context).all():raise ValueError('hold70 schema/finite contract')
    return context


def canonical_action(action):
    result=action.clone();result[:,NULL_COMMANDS]=0
    if result.ndim!=2 or result.shape[-1]!=18 or not torch.isfinite(result).all() or (result.abs()>1+1e-6).any():raise ValueError('bounded native action contract')
    return result
