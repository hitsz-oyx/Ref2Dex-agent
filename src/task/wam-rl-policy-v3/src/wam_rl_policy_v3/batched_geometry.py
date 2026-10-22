"""Device-resident WAM feature and tau construction; fixed 30/60Hz contract."""
import numpy as np
import torch
from .wam_v4_runtime import (GOAL_SEMANTICS, STATIC_THRESHOLD_M,
                             _prepare_goal_inputs, object_poses)


def native_tau(trajectory, rotation):
    """Batched right-hand tau, t=1..8/60, with explicit zero displacement at t=0."""
    values=trajectory[:,:4,0].double()
    previous=torch.cat((torch.zeros_like(values[:,:1]),values[:,:-1]),1)
    result=torch.stack(((previous+values)*.5,values),2).flatten(1,2)
    return torch.einsum('btpi,bji->btpj',result,rotation.double()).float()


def _dynamics(x):
    return torch.cat(((x[:,-1]-x[:,-2])*30,(x[:,-1]-2*x[:,-2]+x[:,-3])*900,x[:,-1]-x[:,0]),-1)


def online_batch(objects,hands,geometries,goal_delta=None,device='cpu',*,
                 goal_world=None,goal_semantics=GOAL_SEMANTICS):
    """Vectorized equivalent of v4_online_samples; float64 geometric arithmetic.

    Goal values and ``object_goal_valid`` are produced by the same shared
    contract helper as the reference/training builder.
    """
    objects_np=np.asarray(objects,np.float64)
    poses_np=object_poses(objects_np)
    poses=torch.as_tensor(poses_np,device=device,dtype=torch.float64)
    hand=torch.as_tensor(hands,device=device,dtype=torch.float64)
    now=poses[:,-1];rotation=now[:,:3,:3];center=now[:,:3,3]
    local_hand=torch.einsum('bhki,bij->bhkj',hand-center[:,None,None],rotation)
    relative=torch.linalg.inv(now)[:,None] @ poses
    flow_np=np.stack([np.asarray(geometry[2]) for geometry in geometries])
    goal_np,goal_valid_np=_prepare_goal_inputs(
        flow_np,poses_np[:,-1,:3,3],poses_np[:,-1,:3,:3],goal_delta=goal_delta,
        goal_world=goal_world,goal_semantics=goal_semantics)
    surface,normals,flow,interaction,interaction_normals=[torch.as_tensor(np.stack(values),device=device) for values in zip(*geometries)]
    moved=torch.einsum('bhij,bpj->bhpi',relative[:,:,:3,:3],surface.double())+relative[:,:,None,:3,3]
    n=len(objects);features=torch.zeros(n,512,18,device=device)
    # The reference computes normal rotation in NumPy float64 then casts once.
    features[:,:,:3]=normals.double() @ relative[:,-1,:3,:3].transpose(1,2)
    features[:,:,3:12]=_dynamics(moved)
    features[:,:,12]=1
    features[:,:,13]=(torch.linalg.vector_norm(features[:,:,9:12],dim=-1)<STATIC_THRESHOLD_M)
    start_hand=torch.zeros(n,2,11,3,device=device);start_hand[:,0]=local_hand[:,-1]
    history=torch.zeros(n,2,11,9,device=device);history[:,0]=_dynamics(local_hand)
    valid=torch.zeros(n,2,11,device=device,dtype=torch.bool);valid[:,0]=True
    return dict(xyz=moved[:,-1].float(),features=features,scene_object=torch.zeros(n,512,device=device,dtype=torch.long),
        point_valid=torch.ones(n,512,device=device,dtype=torch.bool),object_valid=torch.ones(n,1,device=device,dtype=torch.bool),
        scene_points=flow[:,None].float(),start_object_points=flow.float(),goal_object_delta=torch.as_tensor(goal_np,device=device,dtype=torch.float32),
        object_goal_valid=torch.as_tensor(goal_valid_np,device=device,dtype=torch.bool),start_hand=start_hand,start_hand_history=history,start_hand_valid=valid,
        interaction_points=interaction[:,None].float(),interaction_normals=interaction_normals[:,None].float(),
        interaction_query_valid=torch.ones(n,1,128,device=device,dtype=torch.bool))


def workspace_valid(batch,config):
    points=torch.cat((batch['xyz'],batch['start_hand'].flatten(1,2)),1)
    hand_mask=batch['start_hand_valid'].flatten(1) & config.get('hand_in_ptv3',True)
    mask=torch.cat((batch['point_valid'] & (batch['features'][...,14]<.5),hand_mask),1)
    origin=points.new_tensor(config['voxel_origin_m'])
    grid=torch.floor((points-origin)/config['voxel_m'])
    bad=(~torch.isfinite(points)).any(-1) | (grid<0).any(-1) | (grid>65535).any(-1)
    return ~(bad & mask).any(-1) & mask.any(-1)
