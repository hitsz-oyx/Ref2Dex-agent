"""Test the fixed R on actual, unmixed expert recovery hand futures.

CPU loads/validates arrays; the model forward runs on an idle GPU. This is a
privileged diagnostic, never WAM input or autonomous policy evidence.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

TASK=Path(__file__).resolve().parents[2]; ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from audit_wam_ref2_shadow import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=2)
    a=p.parse_args(); source=a.native.resolve(); out=a.output.resolve()
    if ROOT/'outputs/trajectory-policy' not in source.parents or ROOT/'outputs/trajectory-policy' not in out.parents or out.exists():
        raise ValueError('owned source and fresh diagnostic output required')
    m=json.loads((source/'manifest.json').read_text()); audit=json.loads((source/'root_recovery_audit.json').read_text())
    digest=sha(source/'trajectory.npz')
    if (m['status']!='COMPLETED' or not audit['pass'] or digest!=m['trajectory_sha256']
            or digest!=audit['native_data_sha256'] or m['controls']!=192 or m['takeover_tick']!=32 or m['dt']!=1/30):
        raise ValueError('audited 192-control later-takeover source required')
    with np.load(source/'trajectory.npz',allow_pickle=False) as f:
        d={k:f[k].copy() for k in ('q','dq','hand','obj','velocity','pd_targets','action','expert_action','expert_applied','expert_PD')}
    if (not d['expert_applied'][32:,1].all() or d['expert_applied'][:,0].any()
            or not np.array_equal(d['action'][32:,3],d['expert_action'][32:,1])
            or np.abs(d['pd_targets'][32:,3]-d['expert_PD'][32:,1]).max()>1e-6):
        raise ValueError('unmixed actual expert continuation required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from trajectory_policy.pd_retargeter import load_model,decode_target,command
    from trajectory_policy.retargeter_data import features
    torch.set_num_threads(2); start=time.monotonic()
    r_path=ROOT/'outputs/trajectory-policy/competent-pd-fit-20261011-r1/tau.pt'
    if sha(r_path)!='414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f':
        raise ValueError('fixed physically qualified R required')
    model=load_model(torch.load(r_path,map_location='cpu',weights_only=False),'cuda')
    for param in model.parameters(): param.requires_grad_(False)
    offset,scale=[torch.as_tensor(m['native_controller'][key],device='cuda') for key in ('offset','scale')]
    inputs={str(path):sha(path) for path in (Path(__file__),source/'manifest.json',source/'root_recovery_audit.json',source/'trajectory.npz',r_path,TASK/'src/trajectory_policy/pd_retargeter.py',TASK/'src/trajectory_policy/retargeter_data.py')}
    raw={}; summaries={}
    def metrics(pred,label):
        delta=pred-label
        return dict(count=len(delta),xyz_p50_mm=float(np.median(np.linalg.norm(delta[:,:3],axis=-1))*1000),
            xyz_p95_mm=float(np.percentile(np.linalg.norm(delta[:,:3],axis=-1),95)*1000),
            rotation_p95_rad=float(np.percentile(np.abs(delta[:,3:6]).max(-1),95)),
            finger_p95_rad=float(np.percentile(np.abs(delta[:,[6,8,10,12,14,15]]).max(-1),95)))
    ticks=np.arange(32,169)
    for env,name in ((0,'nominal_teacher'),(3,'actual_expert_recovery')):
        future=np.stack([d['hand'][t+np.arange(1,25),env] for t in ticks])
        state,tau=features(*(d[k][ticks,env] for k in ('q','dq','hand','obj','velocity')),future)
        with torch.inference_mode():
            encoded=model(torch.as_tensor(state,device='cuda'),torch.as_tensor(tau,device='cuda'))
            q=torch.as_tensor(d['q'][ticks,env],device='cuda')
            pd=decode_target(encoded,q)
            _,_,clipped=command(pd,q,offset,scale)
            predicted=pd.cpu().numpy()
        true=d['pd_targets'][ticks,env]
        summaries[name]=dict(all=metrics(predicted,true),clipping_rate=float(clipped.float().mean()),blocks={})
        for lo,hi in ((32,48),(48,64),(64,80),(80,128),(128,169)):
            mask=(ticks>=lo)&(ticks<hi)
            summaries[name]['blocks'][f'{lo}:{hi}']=metrics(predicted[mask],true[mask])
        raw.update({name+'_ticks':ticks,name+'_state':state,name+'_tau':tau,name+'_predicted_PD':predicted,name+'_actual_PD':true})
    if any(sha(path)!=h for path,h in inputs.items()) or not all(np.isfinite(v).all() for v in raw.values()):
        raise ValueError('diagnostic source drift/nonfinite output')
    if time.monotonic()-start>60: raise TimeoutError('bounded 60-second diagnostic')
    pids=set(map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()))
    if pids-{os.getpid()}: raise RuntimeError('foreign GPU process')
    out.mkdir(parents=True); np.savez_compressed(out/'predictions.npz',**raw)
    result=dict(status='COMPLETED',panels=summaries,elapsed_s=time.monotonic()-start,input_sha256=inputs,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        scope='privileged actual teacher futures; does not train or execute an actor',
        expert_future_windows=137,source_role=3,expert_controls='every control in [query,query+24) is applied expert',
        mixed_policy_hand_labels_admitted=False,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(out/'result.json',result); print(json.dumps(summaries,indent=2))


if __name__=='__main__': main()
