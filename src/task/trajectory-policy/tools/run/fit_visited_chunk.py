"""Current-policy inverse-control qualification/repair with whole-prefix holdout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/audit')]
from audit_wam_ref2_shadow import write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collections',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=5);p.add_argument('--updates',type=int,default=3000)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/trajectory-policy' not in out.parents or not 1<=a.updates<=3000:raise ValueError('fresh output required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from trajectory_policy.visited_control_data import VisitedControlData,collection_entries
    from trajectory_policy.chunk_retargeter import load_chunk,chunk_windows,SCHEMA
    from trajectory_policy.chunk_objective import control_aware_loss,independent_PD_scale
    from trajectory_policy.retargeter_data import features,ACTIVE
    torch.set_num_threads(2);torch.manual_seed(295);torch.set_float32_matmul_precision('highest')
    entries=collection_entries(a.collections);data=VisitedControlData(ROOT,None,entries)
    if set(data.families.values())!={0,1,2,3}:raise ValueError('all four prefixes required')
    owner=ROOT/'outputs/trajectory-policy';parent=owner/'wam-ref31-own-R-20261013-r1/chunk.pt'
    cp=torch.load(parent,map_location='cpu');net=load_chunk(cp,'cuda')
    scale=independent_PD_scale(torch.tensor(np.array(data.controller['scale'])[list(ACTIVE)],dtype=torch.float32,device='cuda'))
    arrays={name:chunk_windows(data.data['nominal'],envs) for name,envs in [('nominal_train',list(range(16))),('nominal_dev',list(range(16,24)))]}
    arrays['startup_dev']=tuple(v[:128] for v in arrays['nominal_dev'])
    parts={k:[] for k in ('exact_train','continuation_train','contact_train','exact_dev','contact_dev')};queries={}
    for name,takeover in data.takeovers.items():
        d=data.data[name];ticks=np.array(data.valid_ticks[name]);family=data.families[name]
        future=d['hand'][ticks[:,None]+np.arange(1,25),3]
        state,tau=features(*(d[k][ticks,3] for k in ('q','dq','hand','obj','velocity')),future)
        target=d['applied'][ticks[:,None]+np.arange(8),3][...,list(ACTIVE)]
        row=(state,tau,target.astype(np.float32));split='dev' if family==3 else 'train'
        parts['exact_'+split].append(tuple(v[ticks==takeover] for v in row))
        parts['contact_'+split].append(tuple(v[ticks<takeover+16] for v in row))
        if split=='train':parts['continuation_train'].append(tuple(v[ticks>takeover] for v in row))
        queries[name]=dict(family=family,takeover=takeover,split=split,valid_windows=len(ticks),native=str(data.paths[name]),trajectory_sha256=sha(data.paths[name]/'trajectory.npz'))
    for key,rows in parts.items():arrays[key]=tuple(np.concatenate([v[k] for v in rows]) for k in range(3))
    tensors={k:tuple(torch.tensor(v,device='cuda') for v in vs) for k,vs in arrays.items()}
    panels=('nominal_dev','startup_dev','exact_train','contact_train','exact_dev','contact_dev')
    out.mkdir();started=time.monotonic();curve=[]
    np.savez_compressed(out/'panel_inputs.npz',**{name+'_'+field:v for name in panels for field,v in zip(('state','tau','true'),arrays[name])})
    manifest=dict(status='RUNNING',schema=SCHEMA,experiment_id='P-20261013-goal-point-wam-visited-realign',run_id=out.name,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source=str(data.source),parent=str(parent),parent_sha256=sha(parent),source_sha256=sha(data.source/'trajectory.npz'),actor_training_parent=str(owner/'wam-v2-control-fit-20261012-r1/chunk.pt'),collections=list(map(str,a.collections)),queries=queries,excluded=data.excluded,gpu=a.gpu,maximum_updates=a.updates,learning_rate=1e-4,normalization='frozen parent buffers',training_allowed=False)
    write(out/'manifest.json',manifest)
    def evaluate(label):
        net.eval();report={};raw={}
        with torch.no_grad():
            for name in panels:
                s,t,y=tensors[name];prediction=torch.cat([net(s[i:i+256],t[i:i+256]) for i in range(0,len(s),256)])
                diff=(prediction[:,0].clamp(-1,1)-y[:,0])*scale;xyz=diff[:,:3].norm(dim=-1)*1000
                report[name]=dict(n=len(s),xyz_mean_mm=float(xyz.mean()),xyz_p95_mm=float(torch.quantile(xyz,.95)),rotation_p95=float(torch.quantile(diff[:,3:6].abs().amax(-1),.95)),finger_p95=float(torch.quantile(diff[:,6:].abs().amax(-1),.95)),clipping=float((prediction[:,0].abs()>1+1e-6).any(-1).float().mean()))
                raw[name+'_prediction']=prediction.cpu().numpy();raw[name+'_true']=y.cpu().numpy()
        np.savez_compressed(out/(label+'.npz'),**raw);return report
    initial=evaluate('initial');write(out/'initial.json',initial)
    print(json.dumps(dict(initial=initial,excluded=data.excluded)),flush=True)
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-4);gen=torch.Generator(device='cuda').manual_seed(296);best=None;bad=0
    pools=[('nominal_train',np.arange(256)),('nominal_train',np.arange(256,len(arrays['nominal_train'][0]))),('exact_train',np.arange(len(arrays['exact_train'][0]))),('continuation_train',np.arange(len(arrays['continuation_train'][0])))]
    try:
        for step in range(1,a.updates+1):
            batch=[[],[],[]]
            for name,ids in pools:
                selected=torch.tensor(ids,device='cuda')[torch.randint(len(ids),(64,),device='cuda',generator=gen)]
                for k in range(3):batch[k].append(tensors[name][k][selected])
            s,t,y=[torch.cat(v) for v in batch]
            net.train();optimizer.zero_grad(set_to_none=True);loss,_=control_aware_loss(net(s,t),y,net.action_scale,scale)
            if not torch.isfinite(loss):raise FloatingPointError('R loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('R gradient')
            optimizer.step()
            if step%250==0 or step==a.updates:
                report=evaluate('latest');checks={}
                for name in ('nominal_dev','startup_dev','exact_dev','contact_dev'):
                    checks[name]=report[name]['xyz_p95_mm']<=initial[name]['xyz_p95_mm']*1.1+.1
                    checks[name+'/clip']=report[name]['clipping']<=.01
                checks['exact_train']=report['exact_train']['xyz_mean_mm']<=.9*initial['exact_train']['xyz_mean_mm']
                checks['contact_train']=report['contact_train']['xyz_p95_mm']<=.9*initial['contact_train']['xyz_p95_mm']
                for name in panels:
                    for term in ('rotation_p95','finger_p95'):checks[name+'/'+term]=report[name][term]<=initial[name][term]*1.2+.01
                score=sum(report[n]['xyz_mean_mm']/max(initial[n]['xyz_mean_mm'],.1) for n in ('exact_train','contact_train','exact_dev','contact_dev'))/4
                key=(not all(checks.values()),score)
                if best is None or key<best[0]:
                    best=(key,step,report,checks);torch.save(dict(schema=SCHEMA,width=128,model=net.state_dict(),native_controller=data.controller),out/'candidate.pt')
                    (out/'selected.npz').write_bytes((out/'latest.npz').read_bytes())
                elapsed=time.monotonic()-started;gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
                row=dict(step=step,loss=float(loss),reports=report,checks=checks,score=score,elapsed_s=elapsed,gpu=gpu,eta_s=elapsed/step*(a.updates-step));curve.append(row);write(out/'curve.json',dict(curve=curve));print(json.dumps(dict(step=step,gate=all(checks.values()),score=score,elapsed_s=elapsed,gpu=gpu)),flush=True)
                pids=set(map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).split()))
                if pids-{os.getpid()}:raise RuntimeError('foreign GPU process')
                bad=bad+1 if any(report[n]['xyz_p95_mm']>2*initial[n]['xyz_p95_mm']+.1 for n in ('exact_train','exact_dev')) else 0
                if bad>=2:raise RuntimeError('two gross R regressions')
                if elapsed>900:raise TimeoutError('R900s')
        (out/'candidate.pt').rename(out/'chunk.pt')
        result=dict(status='COMPLETED',offline_gate=all(best[3].values()),initial=initial,reports=best[2],checks=best[3],selected_update=best[1],checkpoint_sha256=sha(out/'chunk.pt'),elapsed_s=time.monotonic()-started)
        write(out/'result.json',result);manifest['status']='COMPLETED';write(out/'manifest.json',manifest)
    except BaseException as e:
        manifest.update(status='FAILED',error=repr(e));write(out/'manifest.json',manifest);raise

if __name__=='__main__':main()
