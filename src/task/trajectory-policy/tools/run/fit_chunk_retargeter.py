"""Fixed competent GT-tau->actual native action8 Probe, before any WAM change."""
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
    p.add_argument('--engineering-smoke',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task-owned output required')
    if subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('assigned GPU2 occupied')
    scratch=ROOT/'tmp/wam-v2';scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='2',TMPDIR=str(scratch),OMP_NUM_THREADS='2')
    import torch
    from trajectory_policy.chunk_retargeter import SCHEMA,ChunkRetargeter,chunk_windows,expert_chunk_rows
    from trajectory_policy.retargeter_data import ACTIVE
    torch.set_num_threads(2);torch.manual_seed(295);torch.set_float32_matmul_precision('highest')
    owner=ROOT/'outputs/trajectory-policy';source=owner/'competent-pd-teacher-20261011-r2'
    sm=json.loads((source/'manifest.json').read_text())
    if (sm['status']!='COMPLETED' or sm['training_allowed'] is not True
            or not json.loads((source/'result.json').read_text())['teacher_readiness']
            or not json.loads((source/'audit.json').read_text())['pass']):
        raise ValueError('qualified audited teacher required')
    with np.load(source/'trajectory.npz',allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    arrays={name:chunk_windows(data,envs) for name,envs in
            dict(train=list(range(16)),development=list(range(16,24)),test=list(range(24,32))).items()}
    am_path=owner/'wam-ref2-multitakeover-20261012-r1/manifest.json'
    am=json.loads(am_path.read_text())
    if not json.loads(am_path.with_name('root_aggregation_audit.json').read_text())['pass']:
        raise ValueError('audited actual recovery sources required')
    evidence={str(v.resolve()):sha(v) for v in [Path(__file__),TASK/'src/trajectory_policy/chunk_retargeter.py',
        TASK/'src/trajectory_policy/learned_retargeter.py',TASK/'src/trajectory_policy/retargeter_data.py',
        source/'trajectory.npz',source/'manifest.json',source/'audit.json',am_path]}
    recovery=[]
    for entry in am['admissions']:
        native=Path(entry['native']);nm=json.loads((native/'manifest.json').read_text())
        if (nm['status']!='COMPLETED' or not json.loads((native/'root_recovery_audit.json').read_text())['pass']):
            raise ValueError('completed audited expert source required')
        with np.load(native/'trajectory.npz',allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
        recovery.append(expert_chunk_rows(d,entry['takeover']))
        evidence[str((native/'trajectory.npz').resolve())]=sha(native/'trajectory.npz')
    arrays['recovery_TRAIN']=[np.concatenate([v[k] for v in recovery]) for k in range(3)]
    arrays['train']=[np.concatenate([arrays['train'][k],arrays['recovery_TRAIN'][k]]) for k in range(3)]
    tensors={name:[torch.as_tensor(v,device='cuda') for v in arr] for name,arr in arrays.items()}
    net=ChunkRetargeter(128).cuda()
    with torch.no_grad():
        for name,v,floor in zip(('state','tau','action'),tensors['train'],(.001,.001,.01)):
            getattr(net,name+'_mean').copy_(v.mean(0));getattr(net,name+'_scale').copy_(v.std(0,unbiased=False).clamp_min(floor))
    optimizer=torch.optim.Adam(net.parameters(),lr=3e-4)
    gen=torch.Generator(device='cuda').manual_seed(296)
    out.mkdir();started=time.monotonic();updates=1 if a.engineering_smoke else 3000
    m=dict(schema=SCHEMA,status='RUNNING',task='goal-point-wam',experiment_id='P-20261012-goal-point-wam-ref24-chunk-R',
        run_id=out.name,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        physical_gpu=2,model_seed=295,draw_seed=296,width=128,heads=4,decoder_layers=2,ffn=512,
        source=str(source),recovery_aggregation=str(am_path.parent),input_sha256=evidence,
        batch=256,maximum_updates=updates,loss='all8x12 standardized native action L1',learning_rate=3e-4,
        nominal_sampling='96 startup +96 later',recovery_sampling=64,normalization='TRAIN only',
        target='actual independent native applied action[t:t+7]',inference='replan each tick; only action0 executed',
        development_selection='lowest standardized L1 every250 updates; single final selected packet',
        groups={k:len(v[0]) for k,v in arrays.items()},training_allowed=False)
    write(out/'manifest.json',m)
    x,tau,target=tensors['train'];nominal=8304
    startup=torch.as_tensor(np.flatnonzero(np.repeat(np.arange(519),16)<16),device='cuda')
    later=torch.arange(256,nominal,device='cuda');rec=torch.arange(nominal,len(x),device='cuda')
    best=(float('inf'),None,None);curve=[]
    def predict(name):
        s,t,y=tensors[name]
        with torch.no_grad():return torch.cat([net(s[i:i+512],t[i:i+512]) for i in range(0,len(s),512)])
    try:
        for step in range(1,updates+1):
            ids=torch.cat([pool[torch.randint(len(pool),(count,),generator=gen,device='cuda')]
                           for pool,count in ((startup,96),(later,96),(rec,64))])
            net.train();optimizer.zero_grad(set_to_none=True)
            loss=((net(x[ids],tau[ids])-target[ids])/net.action_scale).abs().mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite chunk loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite chunk gradients')
            optimizer.step()
            if step==1 or step%250==0 or step==updates:
                net.eval();val=float(((predict('development')-tensors['development'][2])/net.action_scale).abs().mean())
                if val<best[0]:best=(val,step,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
                elapsed=time.monotonic()-started
                gpu=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
                row=dict(step=step,loss=float(loss),development_L1=val,elapsed_s=elapsed,eta_s=elapsed/step*(updates-step),gpu=gpu)
                curve.append(row);print(json.dumps(row),flush=True)
                if elapsed>600:raise TimeoutError('chunk fit600s deadline')
                pids=set(map(int,subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader'],text=True).split()))
                if pids-{os.getpid()}:raise RuntimeError('foreign GPU2 process')
        net.load_state_dict(best[2]);net.eval();reports={};raw={}
        scale=np.asarray(sm['native_controller']['scale'])[list(ACTIVE)].copy();scale[6:]/=2
        for name in arrays:
            prediction=predict(name).cpu().numpy();true=arrays[name][2];err=prediction-true
            delta=(prediction[:,0].clip(-1,1)-true[:,0])*scale
            reports[name]=dict(chunk_L1=float((np.abs(err)/net.action_scale.cpu().numpy()).mean()),
                first_native_MAE=float(np.abs(prediction[:,0].clip(-1,1)-true[:,0]).mean()),
                first_PD_xyz_p95_mm=float(np.percentile(np.linalg.norm(delta[:,:3],axis=-1),95)*1000),
                first_PD_rotation_p95_rad=float(np.percentile(np.abs(delta[:,3:6]).max(-1),95)),
                first_PD_finger_p95_rad=float(np.percentile(np.abs(delta[:,6:]).max(-1),95)),
                first_raw_clipping_rate=float((np.abs(prediction[:,0])>1+1e-6).any(-1).mean()))
            raw[name+'_prediction']=prediction;raw[name+'_true']=true
        result=dict(status='COMPLETED',reports=reports,selected_update=best[1],curve=curve,
            elapsed_s=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            conclusion='UNCLEAR; GT execution gate required')
        if not a.engineering_smoke:
            torch.save(dict(schema=SCHEMA,width=128,model=best[2],native_controller=sm['native_controller']),out/'chunk.pt')
            result['checkpoint_sha256']=sha(out/'chunk.pt')
        np.savez_compressed(out/'predictions.npz',**raw);write(out/'result.json',result)
        m.update(status='COMPLETED',result_sha256=sha(out/'result.json'),checkpoint_sha256=result.get('checkpoint_sha256'))
        write(out/'manifest.json',m)
    except BaseException as error:
        m.update(status='FAILED',error=repr(error));write(out/'manifest.json',m);raise


if __name__=='__main__':main()
