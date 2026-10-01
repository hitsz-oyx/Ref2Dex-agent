"""Mesh-support clearance and exact static native PD targets."""
from pathlib import Path
import torch
from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose


def mesh_vertices(path):
    rows=[list(map(float,line.split()[1:4])) for line in Path(path).read_text().splitlines() if line.startswith('v ')]
    values=torch.tensor(rows,dtype=torch.float32)
    if values.ndim!=2 or values.shape[1]!=3 or not torch.isfinite(values).all():
        raise ValueError('mesh vertex schema')
    return values


def table_plane_clearance(object_root,table_root,object_vertices,table_top):
    obj=dexplore_root_pose(object_root);table=dexplore_root_pose(table_root)
    rotation=table[:,:3,:3].transpose(1,2)
    relative=rotation@obj[:,:3,:3]
    translation=(rotation@(obj[:,:3,3]-table[:,:3,3]).unsqueeze(-1)).squeeze(-1)
    heights=torch.einsum('bi,vi->bv',relative[:,2,:],object_vertices)+translation[:,2,None]
    return heights.amin(1)-table_top


def static_pd_action(task,goal):
    action=(goal-task._pd_action_offset)/task._pd_action_scale
    action[:,:6]=(goal[:,:6]-task._pd_action_offset[:6]-task._dof_pos[:,:6])/task._pd_action_scale[:6]
    action[:,6:]=2*action[:,6:]-1
    action[:,[7,9,11,13,16,17]]=0
    if not torch.isfinite(action).all() or (action.abs()>1+1e-6).any():
        raise ValueError('static controller clips')
    action=action.clamp(-1,1)
    predicted=task._action_to_pd_targets(action.clone())
    error=float((predicted-goal).abs().max())
    if error>1e-5:raise ValueError('static PD goal mapping drift')
    return action,error
