"""Recompute saved robot trajectory and first-PD panels from actual own rows."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/audit')]
from audit_competent_pd import independent_features, ACTIVE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=2)
    a=p.parse_args()
    out=a.training.resolve()
    if ROOT/'outputs/goal-point-wam' not in out.parents or (out/'root_panel_audit.json').exists():
        raise ValueError('owned completed training and fresh audit required')
    result=json.loads((out/'result.json').read_text())
    if result['status']!='COMPLETED':raise ValueError('completed adapter training required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],text=True).strip():raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from trajectory_policy.chunk_retargeter import load_chunk
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    packet=torch.load(result['final_checkpoint'],map_location='cpu',weights_only=False)
    manifest=packet['manifest']
    r=load_chunk(torch.load(manifest['C_identity']['qualified_chunk'],map_location='cpu',weights_only=False),'cuda').requires_grad_(False)
    panel=json.loads((out/'panel.json').read_text())
    sources={'nominal':ROOT/'outputs/trajectory-policy/competent-pd-teacher-20261011-r2'}
    sources.update({'recovery%d'%i:Path(path) for i,path in enumerate(manifest['recovery_paths'])})
    traces={}
    for name,path in sources.items():
        with np.load(path/'trajectory.npz') as f:traces[name]={k:f[k] for k in f.files}
    evalstep=packet['stage']*100000+packet['step']
    saved=json.loads((out/('evaluation_step%d.json'%evalstep)).read_text())
    maxerr={};checks=0;oracle_panels={};takeover_queries={}
    def apply_r(state,tau):
        values=[]
        # Preserve the recorded evaluation batch size; TF32 kernels can choose
        # different reductions for a whole panel versus16 rows.
        with torch.no_grad():
            for start in range(0,len(state),16):
                values.append(r(torch.tensor(state[start:start+16],device='cuda',dtype=torch.float32),
                    torch.tensor(tau[start:start+16],device='cuda',dtype=torch.float32))[:,0].cpu().numpy())
        return np.concatenate(values)
    with np.load(out/('panel_step%d.npz'%evalstep)) as arrays:
        for name,keys in panel.items():
            if keys[0][0]=='main':continue
            kind=keys[0][0];d=traces[kind]
            ticks=np.array([k[1] for k in keys]);roles=np.array([k[2] for k in keys])
            future=d['hand'][ticks[:,None]+np.arange(1,25),roles[:,None]].astype(np.float64)
            current=d['hand'][ticks,roles].astype(np.float64)
            obj=d['obj'][ticks,roles].astype(np.float64)
            local_now=(current-obj[:,:3,3][:,None])@obj[:,:3,:3]
            local_future=(future-obj[:,:3,3][:,None,None])@obj[:,:3,:3][:,None]
            order=[0,1,3,5,7,9,2,4,6,8,10]
            truth=local_future[:,:,order].astype(np.float32)-local_now[:,None,order].astype(np.float32)
            tau=arrays[name+'_tau']
            error=np.linalg.norm(tau[:,:,0]-truth,axis=-1)*1000
            for key,count in (('h1_mm',1),('first8_mm',8),('all24_mm',24)):
                raw=error[:,:count].mean(axis=(1,2))
                difference=float(np.abs(raw-arrays[name+'_'+key]).max())
                maxerr[name+'/'+key]=difference
                if difference>3e-4:raise ValueError('own trajectory supervision/metric mismatch: '+name+'/'+key)
                checks+=1
            state,gt_native=independent_features(d,ticks,roles,future)
            native=tau[:,:,0][:,:,[0,1,6,2,7,3,8,4,9,5,10]].reshape(-1,24,33)
            action=apply_r(state,native)
            target=d['applied'][ticks,roles][:,ACTIVE]
            controller=json.loads((sources[kind]/'manifest.json').read_text())['native_controller']
            scale=np.asarray(controller['scale'],np.float32)[list(ACTIVE)].copy();scale[6:]/=2
            diff=(action.clip(-1,1)-target)*scale
            gt_action=apply_r(state,gt_native)
            gt_diff=(gt_action.clip(-1,1)-target)*scale
            oracle_panels[name]=dict(PD_xyz_mm_p95=float(np.percentile(np.linalg.norm(gt_diff[:,:3],axis=-1)*1000,95)),
                clipping_rate=float((np.abs(gt_action)>1+1e-6).any(-1).mean()),
                scope='privileged actual own expert future diagnostic; no actor input or physical qualification')
            raw={'PD_xyz_mm':np.linalg.norm(diff[:,:3],axis=-1)*1000,
                 'PD_rotation_rad':np.linalg.norm(diff[:,3:6],axis=-1),
                 'PD_finger_rad':np.abs(diff[:,6:]).mean(-1),
                 'clipping_rate':(np.abs(action)>1+1e-6).any(-1).astype(float)}
            for key,value in raw.items():
                difference=float(np.abs(value-arrays[name+'_'+key]).max())
                maxerr[name+'/'+key]=difference
                if difference>2e-3:raise ValueError('own applied expert PD/R panel mismatch: '+name+'/'+key+' '+str(difference))
                checks+=1
            if kind.startswith('recovery'):
                takeover_queries[name]=dict(tick=int(ticks[0]),
                    first8_mm=float(arrays[name+'_first8_mm'][0]),
                    first_PD_xyz_mm=float(raw['PD_xyz_mm'][0]),
                    GT_first_PD_xyz_mm=float(np.linalg.norm(gt_diff[0,:3])*1000),
                    scope='exact takeover query; later panel rows are expert continuation states')
                for tick in ticks:
                    if not d['expert_applied'][tick:tick+24,1].all() or d['expert_clipped'][tick:tick+24,1].any():
                        raise ValueError('mixed or clipped recovery future admitted')
    report=dict(status='PASS',checks=checks,metric_max_differences=maxerr,
        final_checkpoint=result['final_checkpoint'],stage=packet['stage'],step=packet['step'],
        frozen_contracts=packet['contracts'],scope='saved robot panels recomputed from own measured expert futures and fixed R; TRAIN recovery, no physical claim')
    report['GT_tau_Chunk_R_panels']=oracle_panels
    report['exact_C_takeover_queries']=takeover_queries
    (out/'root_panel_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
