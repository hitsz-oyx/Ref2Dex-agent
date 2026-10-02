"""Centered native actuation response; physical residuals, no task-score decoder."""
import torch

from .executable_contact_options import INDEPENDENT
from .native_pd_selector import native_pd_targets

TARGET_SCALE = (.1, .1, .1, .005, .005, .005, .002)
RIDGE = .01


def current_inputs(state, palm_position, palm_velocity, hand_force, object_force,
                   mass, gravity, clearance, rest, candidate_pd, probability):
    """No future state or target argument; all proposals precede allocation."""
    n=len(state)
    if candidate_pd.shape!=(n,8,18) or probability.shape!=(n,8):raise ValueError('current proposal dimensions')
    if not torch.allclose(probability.sum(-1),torch.ones(n,device=state.device,dtype=state.dtype)):
        raise ValueError('probabilities do not sum to one')
    relative=state[:,36:39]-palm_position
    relative_velocity=state[:,43:46]-palm_velocity
    weight=mass*gravity
    forces=torch.stack((hand_force.norm(dim=-1).amax(-1),object_force.norm(dim=-1)),-1)
    log_force=torch.log1p(forces/weight[:,None])
    geometry=torch.cat((relative,relative_velocity,log_force,clearance[:,None],
        (state[:,38]-rest).clamp_min(0)[:,None]),-1)
    baseline=torch.cat((state[:,:36],state[:,39:49],relative,log_force,
        clearance[:,None],(state[:,38]-rest).clamp_min(0)[:,None]),-1)
    pd=candidate_pd[:,:,list(INDEPENDENT)]
    mean=(pd*probability[:,:,None]).sum(1)
    centered=pd-mean[:,None]
    return dict(state=baseline,geometry=geometry,centered_candidates=centered,mean_pd=mean)


def rows(record):
    if record['schema']!='ref2dex.contact_geometry_source.v1' or record['future_done'].any():
        raise ValueError('audited no-reset geometry source required')
    # Held targets are never built, including during normalization/qualification.
    keep=record['split_group_bucket']<70
    state=record['state'][keep].double();n=len(state)
    offset,scale=record['pd_offset'].double(),record['pd_scale'].double()
    candidates=native_pd_targets(record['candidate_actions'][keep].double(),state[:,None,:18],offset,scale)
    expected_map=[0,1,2,3,4,5,6,7,0,1]
    if record['allocation_to_option']!=expected_map:raise ValueError('actual randomization map')
    slot_probability=record['allocation_probabilities'][keep].double()
    probability=torch.stack([slot_probability[:,[j for j,a in enumerate(expected_map) if a==k]].sum(-1) for k in range(8)],-1)
    # Serialized float32 probabilities are checked, then normalized to their
    # mathematical sum so the conditional contrast has exact zero expectation.
    probability=probability/probability.sum(-1,keepdim=True)
    chosen=record['assignment'][keep]
    if not torch.allclose(probability[torch.arange(n),chosen],record['propensity'][keep].double(),atol=1e-7):
        raise ValueError('actual propensity drift')
    error=(candidates[torch.arange(n),chosen]-record['actual_pd_targets'][keep,0].double()).abs().max()
    if error>2e-5:raise ValueError('initial actual native PD drift')
    features=current_inputs(state,record['initial_key_positions'][keep,0].double(),
        record['initial_key_velocities'][keep,0].double(),record['initial_hand_force'][keep].double(),
        record['initial_object_force'][keep].double(),record['mass_kg'][keep].double(),record['gravity_magnitude'],
        record['initial_clearance'][keep].double(),record['rest_z'][keep].double(),candidates,probability)
    post=record['future_state'][keep,0].double();dt=record['control_dt_seconds']
    if abs(dt-1/30)>1e-8:raise ValueError('native control period')
    dv=post[:,43:46]-state[:,43:46]
    gravity=torch.tensor([0.,0.,-record['gravity_magnitude']*dt],dtype=torch.float64)
    displacement=post[:,36:39]-state[:,36:39]-state[:,43:46]*dt
    clearance=record['future_clearance'][keep,0].double()-record['initial_clearance'][keep].double()-state[:,45]*dt
    target=torch.cat((dv-gravity,displacement,clearance[:,None]),-1)/torch.tensor(TARGET_SCALE,dtype=torch.float64)
    return dict(**features,target=target,probability=probability,chosen=chosen,
        centered_action=features['centered_candidates'][torch.arange(n),chosen],
        bucket=record['split_group_bucket'][keep],dynamic=dv.norm(dim=-1)>.05,
        cv_velocity_error=dv,gravity_prior=gravity.expand(n,-1),
        initial_native_pd_error=error,episode=[e for e,k in zip(record['episode_id'],keep) if k],
        group=[f'{int(i)}/{int(j)}' for i,j in zip(record['motion_id'][keep],record['start_frame'][keep])],
        total_rows_including_excluded_held=len(keep))


def normalization(data,fit):
    norm={}
    for key in ('state','geometry'):
        x=data[key][fit]
        norm[key+'_mean']=x.mean(0);norm[key+'_std']=x.std(0,unbiased=False).clamp_min(.001)
    norm['action_rms']=data['centered_action'][fit].square().mean(0).sqrt().clamp_min(1e-4)
    return norm


def design(data,norm,actions=None):
    state=((data['state']-norm['state_mean'])/norm['state_std']).clamp(-8,8)
    geometry=((data['geometry']-norm['geometry_mean'])/norm['geometry_std']).clamp(-8,8)
    state=torch.cat((torch.ones_like(state[:,:1]),state),-1)
    phi=torch.cat((torch.ones_like(geometry[:,:1]),geometry),-1)
    action=(data['centered_action'] if actions is None else actions)/norm['action_rms']
    response=torch.einsum('ni,nj->nij',phi,action).flatten(-2)
    return state,phi,response


def ridge(x,y):
    # All coefficients including intercept follow the single frozen penalty.
    eye=torch.eye(x.shape[-1],dtype=x.dtype,device=x.device)
    return torch.linalg.solve(x.T@x/len(x)+RIDGE*eye,x.T@y/len(x))


def fit(data,fit_mask,norm):
    state,_,response=design(data,norm)
    beta=ridge(state[fit_mask],data['target'][fit_mask])
    residual=data['target'][fit_mask]-state[fit_mask]@beta
    cm=ridge(response[fit_mask],residual)
    rng=torch.Generator(device=data['target'].device).manual_seed(16671)
    perm=torch.randperm(int(fit_mask.sum()),generator=rng,device=data['target'].device)
    shuffled_actions=data['centered_action'].clone();shuffled_actions[fit_mask]=shuffled_actions[fit_mask][perm]
    _,_,shuffled_design=design(data,norm,shuffled_actions)
    shuffled=ridge(shuffled_design[fit_mask],residual)
    return dict(state=beta,cm=cm,shuffled=shuffled)


def predict(data,norm,coefficients):
    state,phi,response=design(data,norm)
    baseline=state@coefficients['state']
    predictions=dict(state_only=baseline,cm=baseline+response@coefficients['cm'],
        shuffled=baseline+response@coefficients['shuffled'])
    all_design=torch.einsum('ni,naj->naij',phi,data['centered_candidates']/norm['action_rms']).flatten(-2)
    candidates=baseline[:,None]+all_design@coefficients['cm']
    return predictions,candidates
