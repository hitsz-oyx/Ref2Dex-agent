"""Oracle capacity of scalar mixtures of inertial and rigid hand transports.

Each hand endpoint transports a rigid object. A blended point-flow prediction
is a mixture mean, not necessarily itself a realizable rigid pose. Labels fit
the scalar/endpoint ORACLE; this module does not implement a deployable Cm.
"""
import numpy as np
import torch
from src.task.CmResidual.v118_planner import TorchInspireKinematics
from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose

VISUAL_IDS = (0,1,2,3,4,6,7,9,10,12,13,15,16)
EPSILON_M = 1e-9
BISECTION_STEPS = 52


@torch.no_grad()
def transport_fields(rows, predicted_q, object_local, urdf, device):
    """All computations float64; current/previous state supplies inertia."""
    tensor = lambda value:torch.as_tensor(value, dtype=torch.float64, device=device)
    pose = {name:dexplore_root_pose(tensor(rows[name])) for name in ('current_obj','previous_obj','next_obj','hand_root')}
    local = tensor(object_local)
    points = {name:local[None]@po[:,:3,:3].transpose(-1,-2)+po[:,None,:3,3]
              for name,po in pose.items() if name!='hand_root'}
    current=points['current_obj'];anchor=current-points['previous_obj']
    # Constant WORLD translation and constant relative WORLD rotation increment.
    r=pose['current_obj'][:,:3,:3];old=pose['previous_obj'][:,:3,:3]
    extrapolated_rotation=r@old.transpose(-1,-2)@r
    extrapolated_translation=2*pose['current_obj'][:,:3,3]-pose['previous_obj'][:,:3,3]
    next_inertial=local[None]@extrapolated_rotation.transpose(-1,-2)+extrapolated_translation[:,None]
    state=torch.stack((torch.zeros_like(anchor),next_inertial-current),1)
    fk=TorchInspireKinematics(urdf,device)
    def world(q):return pose['hand_root'][:,None]@fk.forward(tensor(q)[:,None])[:,0]
    hand=world(rows['q']);next_poses={'causal':world(predicted_q),'measured_hand':world(rows['next_q'])}
    inverse=torch.linalg.inv(hand[:,VISUAL_IDS])
    candidates={};deltas={}
    for mode,nxt in next_poses.items():
        delta=nxt[:,VISUAL_IDS]@inverse
        transported=torch.einsum('bnij,bpj->bnpi',delta[:,:,:3,:3],current)+delta[:,:,None,:3,3]
        candidates[mode]=torch.cat((state,transported-current[:,None]),1)
        deltas[mode]=delta
    return dict(anchor=anchor,target=points['next_obj']-current,candidates=candidates,
                deltas=deltas,current_points=current,current_hand_poses=hand,
                measured_next_hand_poses=next_poses['measured_hand'])


@torch.no_grad()
def optimal_segments(anchor, endpoints, target):
    """Minimize smooth mean Euclidean EPE on each convex scalar segment.

    epsilon bounds true/smoothed EPE difference by1e-9m. Global endpoint
    selection follows the TRUE EPE at the solved coefficient. Audit provides
    independent convex optimality certificates and SciPy selected re-solves.
    """
    direction=endpoints-anchor[:,None];offset=anchor[:,None]-target[:,None]
    low=torch.zeros(endpoints.shape[:2],dtype=anchor.dtype,device=anchor.device)
    high=torch.ones_like(low)
    def derivative(coefficient):
        residual=offset+coefficient[...,None,None]*direction
        return ((residual*direction).sum(-1)/(residual.square().sum(-1)+EPSILON_M**2).sqrt()).mean(-1)
    for _ in range(BISECTION_STEPS):
        mid=(low+high)/2;sign=derivative(mid)
        low=torch.where(sign<0,mid,low);high=torch.where(sign>=0,mid,high)
    coefficients=(low+high)/2
    coefficients=torch.where(derivative(torch.zeros_like(low))>=0,torch.zeros_like(low),coefficients)
    coefficients=torch.where(derivative(torch.ones_like(low))<=0,torch.ones_like(low),coefficients)
    all_predictions=anchor[:,None]+coefficients[...,None,None]*direction
    errors=(all_predictions-target[:,None]).norm(dim=-1).mean(-1)
    winner=errors.argmin(1);batch=torch.arange(len(anchor),device=anchor.device)
    return dict(coefficients=coefficients,segment_errors_m=errors,winner=winner,
                prediction=all_predictions[batch,winner])


def episode_report(prediction,target,environments):
    errors=np.linalg.norm(prediction-target,axis=-1).mean(-1)*1000
    parents={str(env):float(errors[environments==env].mean()) for env in np.sort(np.unique(environments))}
    return dict(episode_epe_mm=float(np.mean(list(parents.values()))),window_epe_mm=float(errors.mean()),
                episodes=len(parents),windows=len(errors),per_episode_epe_mm=parents)


def classify(reports,near):
    causal=reports['causal']['episode_epe_mm'];state=reports['state_only']['episode_epe_mm']
    gates=dict(action_capacity=causal<=.9*state,
               near_action_capacity=near['causal']['episode_epe_mm']<=.9*near['state_only']['episode_epe_mm'],
               execution_gap=causal<=1.1*reports['measured_hand']['episode_epe_mm'])
    label='PROMISING' if all(gates.values()) else ('UNCLEAR' if gates['action_capacity'] or gates['near_action_capacity'] else 'UNPROMISING')
    return gates,label
