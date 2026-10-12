"""Bounded original-R diagnostic on the completed geometry-actor native trace.

The enclosing group failed its resource guard; preserve that status. Complete
independently audited native measurements are diagnostic evidence only. Own
expert future is never borrowed into the pure actor or admitted for training.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(TASK/'tools/audit')]
from audit_wam_ref2_shadow import sha, write

SOURCE_SHA = '8513480747187963e2d9b5a16d0dbde2997bcab5bbf5e7e0fb7545d7c1cf8259'
R_SHA = '414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f'


def main():
    source = ROOT/'outputs/trajectory-policy/wam-ref2-execution-geometry-agg-20261012-r1/native'
    out = source/'root_geometry_drift.json'
    if out.exists():
        raise FileExistsError(out)
    m, audit = [json.loads((source/k).read_text()) for k in ('manifest.json','root_recovery_audit.json')]
    group = json.loads((source.parent/'group_status.json').read_text())
    if (m['status']!='COMPLETED' or m['controls']!=192 or m['takeover_tick']!=8
            or m['dt']!=1/30 or sha(source/'trajectory.npz')!=SOURCE_SHA
            or audit['native_data_sha256']!=SOURCE_SHA or audit['pass'] is not True
            or audit['check_count']!=3468 or audit['checks_max']!=0
            or group['status']!='FAILED' or '100MB' not in group['error']):
        raise ValueError('fixed complete native trace and preserved failed resource group required')
    keys=('q','dq','hand','obj','velocity','pd_targets','action','expert_raw',
          'expert_action','expert_PD','expert_applied','expert_clipped','clipped')
    with np.load(source/'trajectory.npz',allow_pickle=False) as f:
        d={k:f[k].copy() for k in keys}
    if (not d['expert_applied'][8:,1].all() or d['expert_applied'][:8,1].any()
            or d['expert_applied'][:,0].any() or d['expert_clipped'][8:,1].any()
            or not np.array_equal(d['action'][8:,3],d['expert_action'][8:,1])
            or not np.array_equal(d['expert_action'][8:,1],np.clip(d['expert_raw'][8:,1],-1,1))
            or not np.array_equal(d['pd_targets'][8:,3],d['expert_PD'][8:,1])):
        raise ValueError('own unclipped applied expert continuation required')
    r_path=ROOT/'outputs/trajectory-policy/competent-pd-fit-20261011-r1/tau.pt'
    if sha(r_path)!=R_SHA:
        raise ValueError('original physically calibrated R required')
    inputs={str(p):sha(p) for p in (Path(__file__), source/'manifest.json',
        source/'root_recovery_audit.json',source/'trajectory.npz',source.parent/'group_status.json',
        r_path,TASK/'src/trajectory_policy/retargeter_data.py',TASK/'src/trajectory_policy/pd_retargeter.py')}
    if subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        raise RuntimeError('GPU2 occupied')
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    import torch
    from trajectory_policy.pd_retargeter import load_model,decode_target,command
    from trajectory_policy.retargeter_data import features
    started=time.monotonic();torch.set_num_threads(2)
    model=load_model(torch.load(r_path,map_location='cpu',weights_only=False),'cuda').eval()
    for p in model.parameters():p.requires_grad_(False)
    offset,scale=[torch.as_tensor(m['native_controller'][k],device='cuda') for k in ('offset','scale')]
    ticks=np.arange(8,169);panels={};raw={}
    for role,name in ((0,'nominal_teacher'),(3,'own_expert_takeover8')):
        future=np.stack([d['hand'][t+np.arange(1,25),role] for t in ticks])
        state,tau=features(*(d[k][ticks,role] for k in ('q','dq','hand','obj','velocity')),future)
        with torch.inference_mode():
            q=torch.as_tensor(d['q'][ticks,role],device='cuda')
            pd=decode_target(model(torch.as_tensor(state,device='cuda'),torch.as_tensor(tau,device='cuda')),q)
            _,_,clipped=command(pd,q,offset,scale)
        prediction=pd.cpu().numpy();label=d['pd_targets'][ticks,role]
        distance=np.linalg.norm((prediction-label)[:,:3],axis=-1)*1000
        blocks={}
        for lo,hi in ((8,16),(16,32),(32,48),(48,64),(64,169)):
            selected=(ticks>=lo)&(ticks<hi)
            blocks[f'{lo}:{hi}']={'count':int(selected.sum()),'XYZ_p95_mm':float(np.percentile(distance[selected],95)),
                'XYZ_mean_mm':float(distance[selected].mean())}
        panels[name]={'XYZ_p95_mm':float(np.percentile(distance,95)),
            'clipping_rate':float(clipped.float().mean()),'blocks':blocks}
        raw[name]={'predicted_XYZ':prediction[:,:3].tolist(),'actual_XYZ':label[:,:3].tolist()}
    actor=np.linalg.norm(d['pd_targets'][:,2,:3]-d['expert_PD'][:,0,:3],axis=-1)*1000
    wrist=np.linalg.norm(d['q'][:,2,:3]-d['q'][:,0,:3],axis=-1)*1000
    earliest=lambda x,threshold: int(np.flatnonzero(x>threshold)[0]) if (x>threshold).any() else None
    prefix=[{'tick':t,'live_expert_PD_XYZ_mm':float(actor[t]),'nominal_wrist_deviation_mm':float(wrist[t]),
        'actor_clipped':bool(d['clipped'][t,2]),'expert_query_clipped':bool(d['expert_clipped'][t,0])}
        for t in (0,1,2,4,8,9,12,16,24,32,48,64)]
    if (any(sha(p)!=h for p,h in inputs.items()) or not all(np.isfinite(v).all() for v in d.values())
            or time.monotonic()-started>60):
        raise RuntimeError('source drift/nonfinite/deadline')
    result={'status':'COMPLETED','engineering_pass':True,'physical_gpu':2,'group_status':'FAILED',
        'group_failure_preserved':group['error'],'source_sha256':SOURCE_SHA,'input_sha256':inputs,
        'panels':panels,'raw_PD_diagnostics':raw,'actor_prefix':prefix,
        'first_actor_live_PD_gt50mm_tick':earliest(actor,50),
        'first_actor_nominal_wrist_gt20mm_tick':earliest(wrist,20),
        'expert_future_training_allowed':False,'mixed_policy_future_allowed':False,
        'scope':'own measured expert future R coverage and new actor early drift; no autonomous benefit or training admission',
        'elapsed_s':time.monotonic()-started,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    write(out,result)
    if out.stat().st_size>300000:
        raise RuntimeError('small diagnostic exceeds300KB')
    print(json.dumps({'panels':panels,'prefix':prefix,'elapsed_s':result['elapsed_s']}))


if __name__=='__main__':main()
