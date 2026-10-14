"""Bounded startup / standardized-input / nearest-neighbor inverse audit."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam.rl_robot import RLRobotWindows
from ptv3_shared_wam.robot import features,ACTIVE
from ptv3_shared_wam.checkpoint import git_commit
from trajectory_policy.chunk_retargeter import ChunkRetargeter


def extract(dataset,indices):
    keys=np.asarray([dataset.keys[int(i)] for i in indices]);n=len(indices)
    result=dict(state=np.zeros((n,87),np.float32),tau=np.zeros((n,24,33),np.float32),
                action=np.zeros((n,12),np.float32),tick=keys[:,0],sequence=keys[:,1])
    for seq in np.unique(keys[:,1]):
        rows=np.flatnonzero(keys[:,1]==seq);t=keys[rows,0];d,_=dataset.sequence(int(seq))
        result['state'][rows],result['tau'][rows]=features(*(d[k][t] for k in ('q','dq','hand','obj','velocity')),
                                      d['hand'][t[:,None]+np.arange(1,25)])
        result['action'][rows]=d['applied'][t][:,list(ACTIVE)]
    return result


def stats(error,tick):
    output={}
    for name,mask in [('all',np.ones(len(error),bool)),('tick0',tick==0),('tick1_15',(tick>0)&(tick<16)),('later',tick>=16)]:
        e=error[mask];output[name]=dict(count=len(e),mean_mm=float(e.mean()) if len(e) else None,p95_mm=float(np.percentile(e,95)) if len(e) else None)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();run=a.run.resolve()
    if out.exists() or out.parent!=run.parent:raise ValueError('fresh sibling Task output required')
    def timeout(*_):raise TimeoutError('120-second audit deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120);torch.set_num_threads(4);start=time.monotonic();out.mkdir()
    manifest=json.loads((run/'manifest.json').read_text());source=Path(manifest['source'])
    train,dev=RLRobotWindows(source,'train'),RLRobotWindows(source,'val')
    with np.load(run/'panels.npz') as f:
        pool=extract(train,f['fit_indices']);panel=extract(dev,f['dev_indices']);fit_ids=f['fit_indices'].copy();dev_ids=f['dev_indices'].copy()
    packet=torch.load(run/'B_control_best.pt',map_location='cpu',weights_only=False);net=ChunkRetargeter(128);net.load_state_dict(packet['model']);net.eval()
    norm={k:getattr(net,k).numpy().copy() for k in ['state_mean','state_scale','tau_mean','tau_scale']}
    def standard(d):
        state=(d['state']-norm['state_mean'])/norm['state_scale'];tau=(d['tau']-norm['tau_mean'])/norm['tau_scale']
        # Fixed equal group contribution; no distance or k chosen on dev.
        return np.concatenate((state/np.sqrt(87),tau.reshape(-1,792)/np.sqrt(792)),axis=-1),state,tau
    x,_,_=standard(pool);y,zstate,ztau=standard(panel)
    with torch.no_grad():
        pred=np.concatenate([net(torch.tensor(panel['state'][i:i+128]),torch.tensor(panel['tau'][i:i+128]))[:,0].numpy() for i in range(0,len(y),128)])
    scale=packet['native_PD_scale'].numpy();err=np.linalg.norm((pred.clip(-1,1)-panel['action'])[:,:3]*scale[:3],axis=-1)*1000
    distance=(y*y).sum(-1)[:,None]+(x*x).sum(-1)[None]-2*y@x.T;distance=np.maximum(distance,0)
    nearest=np.argsort(distance,axis=-1)[:,:8];neighbors=pool['action'][nearest]
    errors={}
    for k in [1,8]:
        estimate=neighbors[:,:k].mean(1);e=np.linalg.norm((estimate-panel['action'])[:,:3]*scale[:3],axis=-1)*1000
        errors[str(k)]=stats(e,panel['tick'])
    ordered=np.argsort(err)[::-1];worst=[]
    for i in ordered[:20]:
        record=dev.records[int(panel['sequence'][i])];neighbor=nearest[i,0];nrecord=train.records[int(pool['sequence'][neighbor])]
        worst.append(dict(sequence=record['sequence'],object=record['object'],tick=int(panel['tick'][i]),error_mm=float(err[i]),
                     teacher_xyz=panel['action'][i,:3].tolist(),pred_xyz=pred[i,:3].tolist(),
                     max_abs_state_z=float(np.abs(zstate[i]).max()),max_abs_tau_z=float(np.abs(ztau[i]).max()),
                     hand_h1_displacement_mm=float(np.linalg.norm(panel['tau'][i,0,:3])*1000),
                     nearest_distance=float(np.sqrt(distance[i,neighbor])),nearest_sequence=nrecord['sequence'],
                     nearest_object=nrecord['object'],nearest_tick=int(pool['tick'][neighbor]),
                     nearest_xyz_error_mm=float(np.linalg.norm(pool['action'][neighbor,:3]-panel['action'][i,:3])*1000)))
    by_object=[]
    for obj in sorted({r['object'] for r in dev.records}):
        mask=np.array([dev.records[int(i)]['object']==obj for i in panel['sequence']]);e=err[mask]
        by_object.append(dict(object=obj,windows=int(mask.sum()),mean_mm=float(e.mean()) if len(e) else None,p95_mm=float(np.percentile(e,95)) if len(e) else None))
    quartiles=[];d1=np.sqrt(distance[np.arange(len(y)),nearest[:,0]])
    for lo,hi in zip([0,.25,.5,.75],[.25,.5,.75,1.]):
        indices=np.argsort(d1)[int(lo*len(y)):int(hi*len(y))];quartiles.append(dict(fraction=[lo,hi],distance_mean=float(d1[indices].mean()),error_mean_mm=float(err[indices].mean())))
    report=dict(status='COMPLETED',git_commit=git_commit(),baseline=str(run),selected_step=packet['step'],device='cpu',elapsed_s=time.monotonic()-start,
                R=stats(err,panel['tick']),nearest_neighbor=errors,worst=worst,
                by_object=sorted(by_object,key=lambda r:r['mean_mm'] if r['windows'] else -1,reverse=True),distance_quartiles=quartiles,
                state_z_p99=float(np.percentile(np.abs(zstate),99)),tau_z_p99=float(np.percentile(np.abs(ztau),99)),
                scope='descriptive fixed-panel audit; no uniqueness or physics claim')
    np.savez_compressed(out/'predictions.npz',prediction=pred,target=panel['action'],errors_mm=err,nearest=nearest,distance=d1,
                        fit_indices=fit_ids,dev_indices=dev_ids)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');signal.alarm(0)
    print(json.dumps({k:v for k,v in report.items() if k not in ['worst','by_object']}));print('worst',json.dumps(worst[:8]))


if __name__=='__main__':main()
