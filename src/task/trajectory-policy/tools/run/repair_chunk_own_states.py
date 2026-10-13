"""Bounded ref3.1 Chunk-R repair: exact C states separated from expert continuation."""
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
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/audit')]
from audit_wam_ref2_shadow import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=3000)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/trajectory-policy' not in out.parents:raise ValueError('fresh owned output required')
    if subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU2 occupied')
    scratch=ROOT/'tmp/ref31';scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='2',TMPDIR=str(scratch),OMP_NUM_THREADS='2')
    import torch
    from trajectory_policy.chunk_retargeter import load_chunk,chunk_windows,expert_chunk_rows,SCHEMA
    from trajectory_policy.chunk_objective import control_aware_loss,independent_PD_scale
    from trajectory_policy.retargeter_data import ACTIVE
    torch.set_num_threads(2);torch.manual_seed(295);torch.set_float32_matmul_precision('highest')
    owner=ROOT/'outputs/trajectory-policy';source=owner/'competent-pd-teacher-20261011-r2'
    parent=owner/'wam-v2-control-fit-20261012-r1/chunk.pt'
    sm=json.loads((source/'manifest.json').read_text())
    if not sm['training_allowed'] or not json.loads((source/'audit.json').read_text())['pass']:raise ValueError('source audit')
    def read(path):
        with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
    data=read(source/'trajectory.npz')
    arrays={name:chunk_windows(data,envs) for name,envs in [('nominal_train',list(range(16))),('nominal_dev',list(range(16,24))),('nominal_report',list(range(24,32)))]}
    arrays['startup_dev']=tuple(v[:16*8] for v in arrays['nominal_dev'])
    inputs={};rec=[]
    for tick in (32,40,48):
        native=owner/f'wam-ref2-execution-C-t{tick}-20261013-r1/native'
        nm=json.loads((native/'manifest.json').read_text())
        if nm['status']!='COMPLETED' or not json.loads((native/'root_recovery_audit.json').read_text())['pass']:raise ValueError('recovery audit')
        rows=expert_chunk_rows(read(native/'trajectory.npz'),tick)
        arrays[f'exact{tick}']=tuple(v[:1] for v in rows)
        arrays[f'contact{tick}']=tuple(v[:16] for v in rows)
        arrays[f'continuation{tick}']=tuple(v[1:] for v in rows)
        rec.append(rows);inputs[str(native)]=sha(native/'trajectory.npz')
    arrays['exact_train']=tuple(np.concatenate([v[k][:1] for v in rec]) for k in range(3))
    arrays['continuation_train']=tuple(np.concatenate([v[k][1:] for v in rec]) for k in range(3))
    tensors={name:tuple(torch.as_tensor(v,device='cuda') for v in row) for name,row in arrays.items()}
    cp=torch.load(parent,map_location='cpu');net=load_chunk(cp,'cuda')
    scale=independent_PD_scale(torch.tensor(np.asarray(sm['native_controller']['scale'])[list(ACTIVE)],dtype=torch.float32,device='cuda'))
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-4)
    gen=torch.Generator(device='cuda').manual_seed(296)
    pools=[('nominal_train',np.arange(256),64),('nominal_train',np.arange(256,len(arrays['nominal_train'][0])),64),('exact_train',np.arange(3),64),('continuation_train',np.arange(192),64)]
    out.mkdir();started=time.monotonic();curve=[];best=None
    manifest=dict(status='RUNNING',schema=SCHEMA,experiment_id='P-20261013-goal-point-wam-ref31-own-R',run_id=out.name,task='goal-point-wam',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source=str(source),parent=str(parent),parent_sha256=sha(parent),recovery_inputs=inputs,physical_gpu=2,seeds=[295,296],maximum_updates=a.updates,batch=256,learning_rate=1e-4,sampling='64 nominal startup +64 nominal later +64 exact C takeover +64 expert continuation',normalization='unchanged parent TRAIN buffers',objective='unchanged control-aware first4 tail1 physicalPD1',training_allowed=False,qualification_directory=str(owner/'wam-ref31-own-R-GT-20261013-r1'))
    write(out/'manifest.json',manifest)
    panels=[k for k in arrays if k not in ('nominal_train','exact_train','continuation_train')]
    def evaluate():
        net.eval();reports={};raw={}
        with torch.no_grad():
            for name in panels:
                s,t,y=tensors[name]
                pred=torch.cat([net(s[i:i+256],t[i:i+256]) for i in range(0,len(s),256)])
                delta=(pred[:,0].clamp(-1,1)-y[:,0])*scale
                xyz=torch.linalg.vector_norm(delta[:,:3],dim=-1)*1000
                reports[name]=dict(n=len(s),PD_xyz_p95_mm=float(torch.quantile(xyz,.95)),PD_xyz_mean_mm=float(xyz.mean()),rotation_p95_rad=float(torch.quantile(delta[:,3:6].abs().amax(-1),.95)),finger_p95_rad=float(torch.quantile(delta[:,6:].abs().amax(-1),.95)),clipping=float((pred[:,0].abs()>1+1e-6).any(-1).float().mean()),chunk_L1=float(((pred-y)/net.action_scale).abs().mean()))
                raw[name+'_prediction']=pred.cpu().numpy();raw[name+'_true']=y.cpu().numpy()
        return reports,raw
    initial,initial_raw=evaluate();np.savez_compressed(out/'initial_predictions.npz',**initial_raw)
    write(out/'initial.json',initial)
    try:
        for step in range(1,a.updates+1):
            batch=[[],[],[]]
            for name,indices,count in pools:
                ids=torch.as_tensor(indices,device='cuda')[torch.randint(len(indices),(count,),device='cuda',generator=gen)]
                for k in range(3):batch[k].append(tensors[name][k][ids])
            s,t,y=[torch.cat(v) for v in batch]
            net.train();optimizer.zero_grad(set_to_none=True)
            loss,_=control_aware_loss(net(s,t),y,net.action_scale,scale)
            if not torch.isfinite(loss):raise FloatingPointError('loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('gradient')
            optimizer.step()
            if step==1 or step%250==0 or step==a.updates:
                reports,raw=evaluate();checks={}
                for name in ('nominal_dev','startup_dev'):
                    for metric,factor,tol in [('PD_xyz_p95_mm',1.1,.1),('rotation_p95_rad',1.2,.01),('finger_p95_rad',1.2,.01),('clipping',1,.01)]:
                        checks[name+'/'+metric]=reports[name][metric]<=initial[name][metric]*factor+tol
                for tick in (32,40,48):
                    for name in (f'exact{tick}',f'contact{tick}'):
                        checks[name+'/PD']=reports[name]['PD_xyz_p95_mm']<=.9*initial[name]['PD_xyz_p95_mm']
                        checks[name+'/clip']=reports[name]['clipping']<=.01
                score=sum(reports[f'exact{t}']['PD_xyz_p95_mm']/initial[f'exact{t}']['PD_xyz_p95_mm']+reports[f'contact{t}']['PD_xyz_p95_mm']/initial[f'contact{t}']['PD_xyz_p95_mm'] for t in (32,40,48))/6
                elapsed=time.monotonic()-started
                gpu=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
                row=dict(step=step,loss=float(loss),reports=reports,checks=checks,pass_gate=all(checks.values()),score=score,elapsed_s=elapsed,eta_s=elapsed/step*(a.updates-step),gpu=gpu)
                curve.append(row);write(out/'curve.json',dict(curve=curve))
                print(json.dumps(dict(step=step,loss=float(loss),gate=all(checks.values()),failed=[k for k,v in checks.items() if not v],score=score,elapsed_s=elapsed,gpu=gpu)),flush=True)
                key=(not all(checks.values()),score)
                if best is None or key<best[0]:
                    best=(key,step,reports,checks)
                    torch.save(dict(schema=SCHEMA,width=128,model=net.state_dict(),native_controller=sm['native_controller']),out/'candidate.pt')
                    np.savez_compressed(out/'predictions.npz',**raw)
                if elapsed>900:raise TimeoutError('900s repair budget')
                pids=set(map(int,subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader'],text=True).split()))
                if pids-{os.getpid()}:raise RuntimeError('foreign GPU2 process')
        (out/'candidate.pt').rename(out/'chunk.pt')
        result=dict(status='COMPLETED',selected_update=best[1],reports=best[2],checks=best[3],offline_gate=all(best[3].values()),initial=initial,checkpoint_sha256=sha(out/'chunk.pt'),elapsed_s=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,conclusion='PROMISING' if all(best[3].values()) else 'UNCLEAR',scope='3 fitted C states,192 expert continuations; not unseen own-state generalization or physical success')
        write(out/'result.json',result);manifest['status']='COMPLETED';write(out/'manifest.json',manifest)
    except BaseException as e:
        manifest.update(status='FAILED',error=repr(e));write(out/'manifest.json',manifest);raise

if __name__=='__main__':main()
