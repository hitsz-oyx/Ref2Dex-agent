"""Reconstruct robot trajectory panels and control metrics from actual traces."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
TP=ROOT/'src/task/trajectory-policy'
sys.path[:0]=[str(TP/'src'),str(TP/'tools/audit')]
from audit_competent_pd import independent_features,independent_pd,ACTIVE
from audit_wam_ref2_shadow import write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',type=int,default=5)
    a=p.parse_args();out=a.run.resolve();m=json.loads((out/'manifest.json').read_text());result=json.loads((out/'result.json').read_text())
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from trajectory_policy.visited_control_data import VisitedControlData,collection_entries
    from trajectory_policy.chunk_retargeter import load_chunk
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    data=VisitedControlData(ROOT,None,collection_entries(m['collections']))
    rpath=Path(m['training_R_checkpoint'])
    if sha(rpath)!=m['training_R_sha256']:raise ValueError('fixed R changed')
    net=load_chunk(torch.load(rpath,map_location='cpu',weights_only=False),'cuda')
    groups=json.loads((out/'panel.json').read_text());keys=json.loads((out/'training_keys.json').read_text())
    for name,rows in keys.items():
        if any(k.startswith('recovery') and data.families[k]==3 for k,t,e in rows):raise ValueError('heldout prefix entered training')
    order=[0,1,6,2,7,3,8,4,9,5,10];maxima={}
    for step in (0,result['selected_update']):
        with np.load(out/f'panel_step{step}.npz') as f:raw={k:f[k].copy() for k in f.files}
        reports=json.loads((out/f'evaluation_step{step}.json').read_text())
        for name,rows in groups.items():
            for metric in ('first8_mm','all24_mm','PD_xyz_mm','clipping'):
                key=name+'_'+metric
                if key in raw:maxima[f'{step}/{key}/mean']=abs(float(raw[key].mean())-reports[name][metric])
            if name in ('oakink2','grab','arctic'):continue
            states=[];targets=[];truth=[]
            for kind,t,e in rows:
                d=data.data[kind];future=d['hand'][t+np.arange(1,25),e]
                state,tau=independent_features(d,np.array([t]),np.array([e]),future[None])
                states.append(state[0]);truth.append(tau.reshape(24,11,3));targets.append(d['applied'][t,e])
            pred=raw[name+'_tau'][:,:,0,order];truth=np.stack(truth)
            error=np.linalg.norm(pred-truth,axis=-1)*1000
            for label,count in [('first8_mm',8),('all24_mm',24)]:maxima[f'{step}/{name}/{label}']=float(np.max(np.abs(error[:,:count].mean((1,2))-raw[name+'_'+label])))
            with torch.no_grad():
                state=torch.tensor(np.stack(states),device='cuda');tau=torch.tensor(pred.reshape(-1,24,33),device='cuda')
                action=net(state,tau)[:,0].cpu().numpy()
            full=np.zeros((len(rows),18),np.float32);full[:,ACTIVE]=action.clip(-1,1)
            delta=independent_pd(full,np.zeros_like(full),data.controller)-independent_pd(np.stack(targets),np.zeros_like(full),data.controller)
            error=np.linalg.norm(delta[:,:3],axis=-1)*1000
            maxima[f'{step}/{name}/PD']=float(np.max(np.abs(error-raw[name+'_PD_xyz_mm'])))
            maxima[f'{step}/{name}/PD_p95']=abs(float(np.percentile(error,95))-reports[name]['PD_p95_mm'])
    curve=json.loads((out/'curve.json').read_text());selected=min(curve,key=lambda x:(not all(x['checks'].values()),x['score']))
    if selected['step']!=result['selected_update'] or all(selected['checks'].values())!=result['offline_gate']:raise ValueError('selection differs')
    packet=torch.load(out/'selected.pt',map_location='cpu',weights_only=False)
    if packet['step']!=selected['step'] or packet['manifest']['training_R_sha256']!=sha(rpath):raise ValueError('pair checkpoint identity')
    passed=all(v<.001 for v in maxima.values())
    write(out/'root_alignment_audit.json',dict(pass_audit=passed,maxima=maxima,whole_prefix_holdout=True,checkpoint_sha256=sha(out/'selected.pt')))
    print(json.dumps(dict(pass_audit=passed,max_error=max(maxima.values()))))
    if not passed:raise ValueError('independent panel tolerance')

if __name__=='__main__':main()
