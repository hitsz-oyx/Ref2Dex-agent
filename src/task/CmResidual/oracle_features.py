"""Saved truth descriptors for the fixed oracle experiment; no terminal labels in features."""
import json
import numpy as np
import torch

DECISION=36
HORIZON=32
ARMS=('state','effect','interaction','joint')

def rotate(q,v):
    return v+2*np.cross(q[...,:3],np.cross(q[...,:3],v)+q[...,3:4]*v)

def inverse(q):
    result=q.copy();result[...,:3]*=-1;return result

def multiply(a,b):
    return np.concatenate((a[...,3:4]*b[...,:3]+b[...,3:4]*a[...,:3]+np.cross(a[...,:3],b[...,:3]),a[...,3:4]*b[...,3:4]-(a[...,:3]*b[...,:3]).sum(-1,keepdims=True)),-1)

def canonical(q):
    return np.where(q[...,3:4]<0,-q,q)

def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)

def descriptor(path,baseline,option):
    trace=load(path/'trace.pt');init=load(path/'initial.pt');base=load(baseline/'trace.pt')
    n=len(init['motion']);end=DECISION+HORIZON
    # Entire scene must share one actual history before its one decision.
    for key in ('object_root','native_q','native_dq','rigid_state'):
        if not torch.equal(trace[key][:DECISION],base[key][:DECISION]):
            raise ValueError('candidate prefix is not exact: '+key)
    common=np.concatenate((base['context'][DECISION].numpy(),np.broadcast_to(option,(n,12))),-1)
    current=base['object_root'][DECISION-1].numpy();future=trace['object_root'][DECISION:end].numpy()
    delta=rotate(inverse(current[None,:,3:7]),future[...,:3]-current[None,:,:3])/.05
    relative_q=canonical(multiply(inverse(current[None,:,3:7]),future[...,3:7]))
    effect=np.concatenate((delta,relative_q,future[...,7:10]/.1,future[...,10:13]),-1).transpose(1,0,2).reshape(n,-1)
    bodies=trace['rigid_state'][DECISION:end,:,:25].numpy();obj=future[:,:,None]
    relative_position=bodies[...,:3]-obj[...,:3];qi=inverse(obj[...,3:7])
    relative_pose=np.concatenate((rotate(qi,relative_position)/.1,canonical(multiply(qi,bodies[...,3:7])),
        rotate(qi,bodies[...,7:10]-obj[...,7:10]-np.cross(obj[...,10:13],relative_position))/.1,
        rotate(qi,bodies[...,10:13]-obj[...,10:13])),-1)
    raw=np.load(path/'contacts.npy',mmap_mode='r');frames=json.loads((path/'contact_frames.json').read_text())
    physics=load(path/'physics_states.pt')['rigid_state'].numpy()
    metadata=json.loads((path/'physical_metadata.json').read_text())
    weight=np.array([r[2]['properties'][0]['mass']*9.81 for r in metadata['actor_body_properties']])
    sums=np.zeros((HORIZON,n,26,15),np.float64)
    for frame in frames:
        tick=frame['tick']-DECISION
        if not 0<=tick<HORIZON:raise ValueError('oracle contacts exceed short horizon')
        env=frame['env'];records=raw[frame['offset']:frame['offset']+frame['count']]
        select=(records['body0']==26)|(records['body1']==26);records=records[select]
        if len(records)==0:continue
        first=records['body0']==26;partner=np.where(first,records['body1'],records['body0']).astype(int)
        if np.any((partner<0)|(partner>25)):raise ValueError('unqualified contact partner')
        lam=records['lambda'].astype(np.float64)
        if np.any(lam < -1e-8):raise ValueError('negative normal force')
        lam=np.maximum(lam,0)
        normal=np.column_stack([records['normal'][c] for c in ('x','y','z')])*np.where(first,1.,-1.)[:,None]
        q=physics[frame['tick']*2+frame['subtick'],env,26,3:7]
        forces=rotate(inverse(q),normal*lam[:,None])
        points=np.column_stack([np.where(first,records['localPos0'][c],records['localPos1'][c]) for c in ('x','y','z')])
        outer=np.stack((points[:,0]**2,points[:,1]**2,points[:,2]**2,points[:,0]*points[:,1],points[:,0]*points[:,2],points[:,1]*points[:,2]),-1)
        values=np.concatenate((np.ones((len(records),1)),(lam>1e-8)[:,None],lam[:,None],forces,points*lam[:,None],outer*lam[:,None]),-1)
        np.add.at(sums[tick,env],partner,values*.5)
    total=sums[...,2:3].copy();sums[...,:2]=np.log1p(sums[...,:2])
    sums[...,2:6]/=weight[None,:,None,None]
    sums[...,6:9]/=np.maximum(total,1e-12)*.05
    sums[...,9:15]/=np.maximum(total,1e-12)*.05**2
    interaction=np.concatenate((relative_pose.transpose(1,0,2,3).reshape(n,-1),sums.transpose(1,0,2,3).reshape(n,-1)),-1)
    features=np.concatenate((common,effect,interaction),-1).astype(np.float32)
    mask=(torch.arange(len(trace['clearance']))[:,None]>=init['phase_stop'][init['motion']][None,:]-74)&(torch.arange(len(trace['clearance']))[:,None]<=init['phase_stop'][init['motion']][None,:]+30)
    good=(trace['object_root'][:,:,2]-init['initial_height'][None,:]>=.03)&(trace['clearance']>=.02)
    if not torch.equal(mask.sum(0),torch.full((n,),105)):raise ValueError('full105 evaluation')
    labels=(good|~mask).all(0).numpy().astype(np.float32)
    if not np.isfinite(features).all():raise ValueError('finite truth descriptor')
    return dict(features=features,labels=labels,motion=init['motion'].numpy(),common_dim=82,effect_dim=effect.shape[1],interaction_dim=interaction.shape[1],raw_contacts=len(raw),prefix_exact=True)

class OracleQ(torch.nn.Module):
    def __init__(self,dim):
        super().__init__();self.network=torch.nn.Sequential(torch.nn.Linear(dim,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
    def forward(self,x):return self.network(x).squeeze(-1)

def masked(x,arm,common,effect):
    x=x.clone()
    if arm in ('state','interaction'):x[:,common:common+effect]=0
    if arm in ('state','effect'):x[:,common+effect:]=0
    return x
