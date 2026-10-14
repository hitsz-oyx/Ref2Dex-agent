"""Measured-state / GT-tau packing without forecasting or future-state inputs."""
import numpy as np
import torch
from ptv3_shared_wam.robot import features,ACTIVE


def pack(dataset,indices):
    keys=np.asarray([dataset.keys[int(i)] for i in indices]);n=len(indices)
    output=dict(state=np.empty((n,87),np.float32),tau=np.empty((n,24,33),np.float32),
                action=np.empty((n,8,12),np.float32),tick=keys[:,0])
    for seq in np.unique(keys[:,1]):
        rows=np.flatnonzero(keys[:,1]==seq);ticks=keys[rows,0];d,_=dataset.sequence(int(seq))
        output['state'][rows],output['tau'][rows]=features(*(d[k][ticks] for k in ('q','dq','hand','obj','velocity')),
                                                        d['hand'][ticks[:,None]+np.arange(1,25)])
        output['action'][rows]=d['applied'][ticks[:,None]+np.arange(8)][...,list(ACTIVE)]
    return output


def tensors(panel):
    return {k:torch.from_numpy(v) if k!='tick' else v for k,v in panel.items()}


@torch.no_grad()
def evaluate(net,panel,pdscale):
    net.eval()
    prediction=torch.cat([net(panel['state'][i:i+128],panel['tau'][i:i+128]) for i in range(0,len(panel['state']),128)])
    delta=(prediction[:,0].clamp(-1,1)-panel['action'][:,0])*pdscale
    xyz=delta[:,:3].norm(dim=-1).numpy()*1000;tick=panel['tick']
    result=dict(xyz_mean_mm=float(xyz.mean()),xyz_p95_mm=float(np.percentile(xyz,95)),
                rotation_p95_rad=float(np.percentile(delta[:,3:6].abs().amax(-1).numpy(),95)),
                finger_p95_rad=float(np.percentile(delta[:,6:].abs().amax(-1).numpy(),95)),
                clipping=float((prediction[:,0].abs()>1).any(-1).float().mean()),
                raw_first_MAE=float((prediction[:,0]-panel['action'][:,0]).abs().mean()))
    for name,mask in [('startup',tick<16),('later',tick>=16)]:
        e=xyz[mask];result[name]=dict(count=len(e),xyz_mean_mm=float(e.mean()) if len(e) else None,
                                    xyz_p95_mm=float(np.percentile(e,95)) if len(e) else None)
    net.train();return result
