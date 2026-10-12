"""Optimize independent tau variables to screen a frozen R's local PD fit.

This is an unconstrained privileged diagnostic, never an executable actor or
a physically valid trajectory claim. Failure is not proof of global infeasibility.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from audit_wam_ref2_shadow import sha, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=2)
    args=parser.parse_args(); out=args.output.resolve()
    if out.exists() or ROOT/'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task-owned audit output required')
    if subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    import torch.nn.functional as F
    from trajectory_policy.pd_retargeter import load_model,encode_target,decode_target
    torch.set_num_threads(2); started=time.monotonic()
    fit,native=args.fit.resolve(),args.native.resolve()
    fm=json.loads((fit/'manifest.json').read_text()); nm=json.loads((native/'manifest.json').read_text())
    if fm['status']!='COMPLETED' or nm['status']!='COMPLETED' or not json.loads((native/'root_recovery_audit.json').read_text())['pass']:
        raise ValueError('completed audited inputs required')
    if sha(fit/'predictions.npz')!=fm['predictions_sha256'] or sha(native/'trajectory.npz')!=nm['trajectory_sha256']:
        raise ValueError('source array identity mismatch')
    with np.load(fit/'predictions.npz',allow_pickle=False) as f:
        arrays={k:f[k].copy() for k in f.files}
    with np.load(native/'trajectory.npz',allow_pickle=False) as f:
        data={k:f[k].copy() for k in ('q','learner_state','expert_PD')}
    rpath=ROOT/'outputs/trajectory-policy/competent-pd-fit-20261011-r1/tau.pt'
    if sha(rpath)!='414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f':
        raise ValueError('fixed physically calibrated R required')
    r=load_model(torch.load(rpath,map_location='cpu',weights_only=False),'cuda')
    for p in r.parameters(): p.requires_grad_(False)
    out.mkdir(parents=True)
    inputs={str(p):sha(p) for p in (Path(__file__),fit/'predictions.npz',fit/'manifest.json',native/'trajectory.npz',native/'manifest.json',native/'root_recovery_audit.json',rpath)}
    result=dict(scope='unconstrained per-state tau inversion, not actor or physical-trajectory evidence',panels={})
    packed={}
    bounds_low=(r.target_mean+r.target_scale*(r.net[-1].bias-r.net[-1].weight.abs().sum(-1))).detach().cpu().numpy()
    bounds_high=(r.target_mean+r.target_scale*(r.net[-1].bias+r.net[-1].weight.abs().sum(-1))).detach().cpu().numpy()
    for panel in ('query_training','query_validation'):
        ids=arrays['final_'+panel+'_row_ids']; selected=arrays['rows'][ids]; ticks,roles=selected[:,1],selected[:,2]
        q=torch.as_tensor(data['q'][ticks,roles],device='cuda')
        state=torch.as_tensor(data['learner_state'][ticks,roles-2],device='cuda')
        true_pd=data['expert_PD'][ticks,roles-2]
        y_np=encode_target(true_pd,data['q'][ticks,roles]); y=torch.as_tensor(y_np,device='cuda')
        tau0=torch.as_tensor(arrays['final_'+panel+'_tau'],device='cuda')
        tau=torch.nn.Parameter(tau0.clone()); optimizer=torch.optim.Adam([tau],lr=.01)
        def metrics():
            with torch.no_grad():
                encoded=r(state,tau); actual=decode_target(encoded,q)
                delta=(actual-torch.as_tensor(true_pd,device='cuda')).cpu().numpy()
                return dict(xyz_p95_mm=float(np.percentile(np.linalg.norm(delta[:,:3],axis=-1),95)*1000),
                    rotation_p95_rad=float(np.percentile(np.abs(delta[:,3:6]).max(-1),95)),
                    finger_p95_rad=float(np.percentile(np.abs(delta[:,[6,8,10,12,14,15]]).max(-1),95))),actual.cpu().numpy()
        initial,_=metrics(); curve=[]
        for step in range(1,101):
            optimizer.zero_grad(set_to_none=True)
            loss=F.huber_loss(r(state,tau)/r.target_scale,y/r.target_scale)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite inverse diagnostic')
            loss.backward()
            if step==1:
                initial_grad=tau.grad.abs().flatten(1).sum(-1).detach().cpu().numpy()
            optimizer.step()
            if step==1 or step%25==0:
                pids=set(map(int,subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()))
                if pids-{os.getpid()}: raise RuntimeError('foreign GPU PID')
                curve.append(dict(step=step,loss=float(loss)))
            if time.monotonic()-started>60: raise TimeoutError('60s inverse audit bound')
        final,pd=metrics()
        movement=torch.linalg.vector_norm((tau.detach()-tau0).reshape(-1,24,11,3),dim=-1).cpu().numpy()
        violation=np.maximum(np.maximum(bounds_low-y_np,y_np-bounds_high),0)
        result['panels'][panel]=dict(initial=initial,final=final,curve=curve,
            tau_change_point_p95_mm=float(np.percentile(movement,95)*1000),
            zero_initial_gradient_rows=int((initial_grad==0).sum()),
            expert_rows_outside_global_R_bound=int((violation>1e-6).any(-1).sum()),
            caveat='unrestricted tau may be physically implausible; local optimizer failure is inconclusive')
        packed.update({panel+'_optimized_tau':tau.detach().cpu().numpy(),panel+'_optimized_pd':pd,
                       panel+'_true_pd':true_pd,panel+'_rows':selected})
    if any(sha(p)!=h for p,h in inputs.items()): raise ValueError('source drift')
    np.savez_compressed(out/'predictions.npz',**packed)
    result.update(elapsed_s=time.monotonic()-started,input_sha256=inputs,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(out/'result.json',result); print(json.dumps(result['panels'],indent=2))


if __name__=='__main__': main()
