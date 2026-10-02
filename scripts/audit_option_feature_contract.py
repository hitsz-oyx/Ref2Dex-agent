"""Raw PRE observation and Torch64 SDK transform reconstruction, no future labels."""
import numpy as np
import torch
from scripts.audit_observed_support_value import matrices


def independent_points(initial,trace,metadata,base,ticks):
    n=lambda k:initial[k].numpy();v=lambda k:trace[k].numpy();env=np.arange(768);motion=n('motion');stop=n('phase_stop')[motion]
    q=np.concatenate((n('base_q')[None],v('native_q')[:-1]),0);dq=np.concatenate((n('initial_dof_vel')[None],v('native_dq')[:-1]),0);obj=np.concatenate((n('object_root')[None],v('object_root')[:-1]),0);contact=np.concatenate((n('initial_contact')[None],v('contact')[:-1]),0)
    progress=np.minimum(ticks+1,stop);ref=n('native_reference_q')[motion,progress]
    context=np.concatenate((q[ticks,env],dq[ticks,env],obj[ticks,env],contact[ticks,env],ref,(progress/stop).astype(np.float32)[:,None]),-1).astype(np.float32)
    assert np.max(np.abs(context-v('context')[ticks,env]))<=1e-6
    compact=np.concatenate((np.clip((context-base['observation_mean'].numpy())/base['observation_std'].numpy(),-10,10),np.ones((768,1),np.float32),((stop+30-ticks)/np.float32(202))[:,None]),-1).astype(np.float32)
    current=torch.from_numpy(obj[ticks,env]).double();older=trace['object_root'][ticks-2,env].double();inverse=matrices(current[:,3:7]).transpose(-1,-2);previous_inverse=matrices(older[:,3:7]).transpose(-1,-2)
    pos=torch.matmul(inverse[:,None],(trace['hand_body_position'][ticks-1,env].double()-current[:,None,:3])[...,None]).squeeze(-1)
    oldpos=torch.matmul(previous_inverse[:,None],(trace['hand_body_position'][ticks-2,env].double()-older[:,None,:3])[...,None]).squeeze(-1)
    rot=torch.matmul(inverse[:,None],matrices(trace['hand_body_quaternion'][ticks-1,env]))
    force=torch.cat((trace['object_force'][ticks-1,env,None],trace['hand_force'][ticks-1,env]),1).double();weight=torch.tensor([p['mass'] for p in metadata['object_body_properties']],dtype=torch.float64)*torch.linalg.vector_norm(torch.tensor(metadata['gravity'],dtype=torch.float64))
    force=torch.matmul(inverse[:,None],force[...,None]).squeeze(-1)/weight[:,None,None]
    prior=torch.stack((current[:,2]-older[:,2],(trace['clearance'][ticks-1,env]-trace['clearance'][ticks-2,env]).double()),-1)/.005
    extra=torch.cat((pos.reshape(-1,15),rot[...,:2].reshape(-1,30),force.reshape(-1,18),prior,(pos-oldpos).reshape(-1,15)),-1).float().numpy()
    return compact,extra
