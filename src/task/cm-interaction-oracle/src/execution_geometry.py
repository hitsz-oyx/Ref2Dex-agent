"""Separate decision-time execution prediction from realized-motion oracles."""
import torch
from geometric_consequence import standardize_fit, normalize, ridge_fit, ridge_predict
from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose
from src.task.CmResidual.v118_planner import QUERY_LINKS


def joint_units(device):
    # Native q0:3 metres; q3:18 radians. Fixed feature/target conventions.
    return torch.tensor([.02]*3+[.1]*3+[.32]*12,device=device)


def execution_inputs(p, h, geometry, fit_ids, norm=None):
    """Input whitelist: current history and nominal source target only."""
    q=p['history'][:,-1,:18].to(h.device)
    velocity=p['history'][:,-1,18:36].to(h.device)
    raw=torch.cat((q,velocity,geometry['targets'][:,0]),-1)
    norm=standardize_fit(raw,fit_ids) if norm is None else norm
    state=torch.cat((h,normalize(raw,norm)),-1)
    action=geometry['joint']/joint_units(h.device)
    return state,action,norm


def fit_execution(state, action, target, actual_arms, fit_ids, use_action):
    rows=torch.arange(len(state),device=state.device)
    a=action[rows,actual_arms] if use_action else torch.zeros_like(action[:,0])
    model=ridge_fit(torch.cat((state,a),-1),target,fit_ids)
    candidates=[]
    for arm in range(15):
        a=action[:,arm] if use_action else torch.zeros_like(action[:,0])
        candidates.append(ridge_predict(torch.cat((state,a),-1),model))
    return model,torch.stack(candidates,1)


@torch.no_grad()
def endpoint_flows(p, bridge, geometry, candidate_q):
    """FK predicted or oracle q in the CURRENT object frame.

    Reads no future object pose, measured future root, force or E/I. The
    fixed actor root is independently checked against all actual step8 tips.
    """
    if candidate_q.ndim!=3 or candidate_q.shape[-1]!=18:
        raise ValueError('candidate q must be [windows,candidates,18]')
    root=dexplore_root_pose(p['hand_root'].to(bridge.device))
    obj=dexplore_root_pose(p['before'][:,:13].to(bridge.device))
    links=root[:,None,None]@bridge.kinematics.forward(candidate_q)
    world,_=bridge.points(links)
    points=torch.einsum('bknj,bji->bkni',world-obj[:,None,None,:3,3],obj[:,:3,:3])
    flow=points-geometry['points'][:,None]
    return flow,links


def verify_realized_fk(p, links):
    tips=[QUERY_LINKS.index(n+'_tip') for n in ('index','middle','pinky','ring','thumb')]
    tip_error=(links[:,0,tips,:3,3]-p['fingertip_positions'][:,7].to(links.device)).norm(dim=-1)
    measured=dexplore_root_pose(torch.nn.functional.pad(p['hand_base_pose'][:,7].to(links.device),(0,6)))
    base_error=(links[:,0,0]-measured).abs().max()
    if tip_error.max()>=1e-4 or base_error>=1e-4:
        raise ValueError('fixed-root realized FK/live mismatch')
    return dict(tip_error_max_m=float(tip_error.max()),handbase_matrix_error=float(base_error))


def spatial_variant(geometry, flow):
    result=dict(geometry)
    result['nominal_flow']=flow
    return result


def joint_slots(delta):
    return torch.nn.functional.pad(delta/joint_units(delta.device),(0,14))
