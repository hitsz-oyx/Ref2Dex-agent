"""Cross old/new R on identical retained GT and frozen-actor contact queries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/audit')]
from audit_competent_pd import independent_features,ACTIVE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True);p.add_argument('--gpu',type=int,default=4)
    a=p.parse_args();destination=a.fit/'frozen_actor_query_cross.json'
    if destination.exists():raise FileExistsError(destination)
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from trajectory_policy.chunk_retargeter import load_chunk
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    directory=ROOT/'outputs/goal-point-wam/wam-control-context-20261013-r3'
    actor=torch.load(directory/'stage2_step4500.pt',map_location='cpu',weights_only=False)
    manifest=actor['manifest'];panels=json.loads((directory/'panel.json').read_text())
    models={name:load_chunk(torch.load(path,map_location='cpu',weights_only=False),'cuda') for name,path in [('old',Path(manifest['C_identity']['qualified_chunk'])),('repaired',a.fit/'chunk.pt')]}
    reports={}
    with np.load(directory/'panel_step204500.npz') as saved:
        for name,keys in panels.items():
            kind=keys[0][0]
            if not kind.startswith('recovery'):continue
            source=Path(manifest['recovery_paths'][int(kind[-1])]);m=json.loads((source/'manifest.json').read_text())
            with np.load(source/'trajectory.npz') as f:d={k:f[k] for k in f.files}
            ticks=np.array([k[1] for k in keys]);envs=np.array([k[2] for k in keys])
            future=d['hand'][ticks[:,None]+np.arange(1,25),envs[:,None]]
            state,gt=independent_features(d,ticks,envs,future)
            predicted=saved[name+'_tau'][:,:,0][:,:,[0,1,6,2,7,3,8,4,9,5,10]].reshape(-1,24,33)
            target=d['applied'][ticks,envs][:,ACTIVE];scale=np.array(m['native_controller']['scale'])[ACTIVE];scale[6:]/=2
            result={}
            for r,model in models.items():
                for input_name,tau in [('GT',gt),('actor',predicted)]:
                    with torch.no_grad():commands=model(torch.tensor(state,dtype=torch.float32,device='cuda'),torch.tensor(tau,dtype=torch.float32,device='cuda'))[:,0].cpu().numpy()
                    err=np.linalg.norm((commands.clip(-1,1)-target)[:,:3]*scale[:3],axis=-1)*1000
                    result[r+'_'+input_name]=dict(exact_PD_mm=float(err[0]),contact16_PD_p95_mm=float(np.percentile(err,95)),clipping=float((np.abs(commands)>1+1e-6).any(-1).mean()))
            reports[name]=dict(exact_tick=int(ticks[0]),n=len(ticks),cross=result)
    destination.write_text(json.dumps(dict(reports=reports,scope='identical fixed actor/GT query arrays and own states; fitted contact panels, no new rollouts or adaptation'),indent=2)+'\n')
    print(json.dumps(reports))

if __name__=='__main__':main()
