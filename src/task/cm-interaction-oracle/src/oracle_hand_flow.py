"""Ref11 oracle hand trajectories: planning input separated from execution."""
from hashlib import sha256
import numpy as np
import torch
from torch import nn
from geometric_consequence import standardize_fit,normalize,pca_fit,pca_apply
from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose
from src.task.CmResidual.v118_planner import QUERY_LINKS

FLOW_WIDTH=720


def temporal_actions(points):
    """Corresponding points at0/4/8, all in the CURRENT object frame."""
    if points.ndim!=4 or points.shape[1:]!=(3,120,3):
        raise ValueError('expected [windows,3,120,3] at0/4/8')
    first=points[:,1]-points[:,0];second=points[:,2]-points[:,1]
    endpoint=points[:,2]-points[:,0]
    return dict(State=torch.zeros(len(points),FLOW_WIDTH,device=points.device),
                Endpoint=torch.nn.functional.pad(endpoint.flatten(1)/.02,(0,360)),
                Chunk=torch.cat((first.flatten(1),second.flatten(1)),-1)/.02)


@torch.no_grad()
def measured_flow_inputs(p,bridge):
    """FK actual q at4/8; future object, contact, actions and labels forbidden."""
    dev=bridge.device
    q=torch.stack((p['history'][:,-1,:18],p['native_q'][:,3],p['native_q'][:,7]),1).to(dev)
    root=dexplore_root_pose(p['hand_root'].to(dev));obj=dexplore_root_pose(p['before'][:,:13].to(dev))
    points,normals,errors=[],[],{}
    tips=[QUERY_LINKS.index(name+'_tip') for name in ('index','middle','pinky','ring','thumb')]
    for ids in torch.arange(len(q),device=dev).split(64):
        links=root[ids,None,None]@bridge.kinematics.forward(q[ids])
        world,normal=bridge.points(links)
        points.append(torch.einsum('bknj,bji->bkni',world-obj[ids,None,None,:3,3],obj[ids,:3,:3]))
        normals.append(torch.einsum('bnj,bji->bni',normal[:,0],obj[ids,:3,:3]))
        cpu_ids=ids.cpu()
        for at,step in enumerate((0,4,8)):
            tip=p['before_fingertip_positions'][cpu_ids] if not step else p['fingertip_positions'][cpu_ids,step-1]
            base=p['before_hand_base_pose'][cpu_ids] if not step else p['hand_base_pose'][cpu_ids,step-1]
            measured=dexplore_root_pose(torch.nn.functional.pad(base.to(dev),(0,6)))
            tip_error=float((links[:,at,tips,:3,3]-tip.to(dev)).norm(dim=-1).max())
            base_error=float((links[:,at,0]-measured).abs().max())
            errors['tip_step'+str(step)]=max(errors.get('tip_step'+str(step),0),tip_error)
            errors['base_step'+str(step)]=max(errors.get('base_step'+str(step),0),base_error)
    if max(errors.values())>=1e-4:raise ValueError('oracle FK/live trajectory mismatch: '+str(errors))
    points=torch.cat(points);normals=torch.cat(normals)
    geometry=torch.cat((points[:,0].flatten(1)/.1,normals.flatten(1)),-1)
    actions=temporal_actions(points)
    errors['chunk_endpoint_identity']=float((actions['Chunk'][:,:360]+actions['Chunk'][:,360:]-actions['Endpoint'][:,:360]).abs().max())
    assert errors['chunk_endpoint_identity']<1e-5
    return geometry,actions,errors


def current_state(p,geometry,train,norms=None):
    """Only current physical/q/dq history; no base/PD/actions/execution model."""
    dev=geometry.device
    phys=p['before'].to(dev).clone()
    origin=phys[:,:3].clone()
    phys[:,13:48].reshape(-1,5,7)[:,:,:3]-=origin[:,None]
    phys[:,:3]=0 # remove irrelevant absolute environment translation
    history=p['history'][:,:,:36].to(dev).flatten(1)
    current=p['history'][:,-1,:36].to(dev)
    if norms is None:
        norms=dict(physical=standardize_fit(phys,train),current=standardize_fit(current,train),
                   history=pca_fit(history,train,32),geometry=pca_fit(geometry,train,16))
    h=torch.cat((normalize(phys,norms['physical']),normalize(current,norms['current']),
                 pca_apply(history,norms['history']),pca_apply(geometry,norms['geometry'])),-1)
    return h,norms


def exact_observable_groups(p,ids):
    """Evidence support only: equal observed state is not simulator restoration."""
    rows=torch.cat((p['before'],p['history'].flatten(1),p['hand_root'],
                    p['before_fingertip_positions'].flatten(1),p['before_hand_base_pose']),-1).contiguous().numpy()
    groups={}
    for i in ids:groups.setdefault(sha256(rows[i].tobytes()).hexdigest(),[]).append(int(i))
    pairs=[(g[a],g[b]) for g in groups.values() for a in range(len(g)) for b in range(a+1,len(g))]
    return np.asarray(pairs,dtype=np.int64).reshape(-1,2),dict(exact_groups_with_repeats=sum(len(g)>1 for g in groups.values()),
        exact_observable_pairs=len(pairs),scope='Observed state equality only; no restored simulator/hidden-contact-state guarantee.')


class OracleFlowHead(nn.Module):
    def __init__(self,h_width):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(h_width+FLOW_WIDTH,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,26))

    def forward(self,h,flow):return self.net(torch.cat((h,flow),-1))


@torch.no_grad()
def evaluate(model,h,flow):
    return torch.cat([model(h[ids],flow[ids]) for ids in torch.arange(len(h),device=h.device).split(64)])


def fit(h,flow,target,train,updates=300):
    torch.manual_seed(255);model=OracleFlowHead(h.shape[1]).to(h.device)
    initial=sha256(torch.cat([v.detach().flatten() for v in model.parameters()]).cpu().numpy().tobytes()).hexdigest()
    generator=torch.Generator(device=h.device).manual_seed(256)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    losses=[]
    for _ in range(updates):
        ids=train[torch.randint(len(train),(64,),generator=generator,device=h.device)]
        loss=(model(h[ids],flow[ids])-target[ids]).square().mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite oracle flow loss')
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
        losses.append(float(loss.detach()))
    model.eval();value=evaluate(model,h,flow)
    if not torch.isfinite(value).all():raise ValueError('nonfinite oracle flow prediction')
    return model,value,dict(initial_hash=initial,parameters=sum(v.numel() for v in model.parameters()),updates=updates,
        losses=losses,source_mse=float((value[train]-target[train]).square().mean()))
