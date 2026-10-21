"""Capacity-tune and supervise user-requested GPUs4/5/6 WAM continuation."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import shutil
import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.local_scale import transition_local
from ptv3_shared_wam_v6.local_subset import EXPANDED_SCOPE
from launch_local_training import read,write,gpu_state,released,alive


def audit(path,world,batch,step,normalizer):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if (saved['step']!=step or saved['world_size']!=world
            or saved['execution_contract']['effective_global_batch']!=world*batch
            or len(saved['random_state'])!=world):raise ValueError('scaled checkpoint binding mismatch')
    steps=set(int(v['step']) for v in saved['optimizer']['state'].values())
    if steps!={step} or len(saved['optimizer']['state'])!=568:raise ValueError('optimizer step/state mismatch')
    def finite(value):
        if isinstance(value,torch.Tensor):
            if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('nonfinite checkpoint')
        elif isinstance(value,dict):
            for child in value.values():finite(child)
        elif isinstance(value,(list,tuple)):
            for child in value:finite(child)
    finite(saved['model']);finite(saved['optimizer'])
    for key,value in normalizer.items():
        if not torch.equal(saved['model'][key],value):raise ValueError('normalization changed')
    result=dict(status='PASS',step=step,world_size=world,global_batch=world*batch,
                optimizer_states=568,optimizer_steps=sorted(steps),rng_states=world,bytes=path.stat().st_size)
    del saved;gc.collect();return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','previous-recipe','recipe','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--git-commit',required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);out=a.output.resolve();(out/'inputs').mkdir()
    torch.set_num_threads(2)
    state=dict(status='PREFLIGHT',supervisor_pid=os.getpid(),git_commit=a.git_commit,
        work_version='v6.local-expanded-ddp-supervisor.v1',run_id=out.name,physical_gpus=[4,5,6],
        resource_authorization='User explicitly requested multi-GPU scale-up after the single-GPU plan; uses the three released own PPO GPUs for this run',
        stages=[],started_at=time.time(),checkpoint=str(a.checkpoint.resolve()))
    def publish():write(out/'process.json',state)
    publish();released([4,5,6])
    preflight=read(a.recipe.parent/'preflight.json')
    if preflight['status']!='PASS':raise ValueError('expanded reader preflight required')
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if source['world_size']!=1:raise ValueError('preserved single-GPU parent required')
    normalizer={k:v.clone() for k,v in source['model'].items() if k.startswith('normalizer.')}
    parent_step=source['step'];state['source_step']=parent_step
    base=dict(source['config'],local_subset_scope=EXPANDED_SCOPE,updates=10000,
        validation_interval=500,checkpoint_interval=500,validation_initial_step=0,
        validation_before_training=False,log_interval=20)
    root=Path(os.environ.get('REF2DEX_RUN_ROOT',paths.ROOT))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',NCCL_P2P_DISABLE='1',
        NCCL_DEBUG='WARN',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',NCCL_ASYNC_ERROR_HANDLING='1',
        TMPDIR=str(out),TRITON_CACHE_DIR=str(out/'triton'),TORCH_EXTENSIONS_DIR=str(out/'extensions'))
    requested=[];child=None;current=None
    signal.signal(signal.SIGTERM,lambda *_:requested.append('SIGTERM'))
    signal.signal(signal.SIGINT,lambda *_:requested.append('SIGINT'))
    def run(name,command,environment,limit):
        nonlocal child,current
        current=out/name;record=dict(name=name,command=command,started_at=time.time())
        state['stages'].append(record);state.update(status='RUNNING',phase=name);publish()
        with (out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['pid']=child.pid;publish();start=time.monotonic();deadline=None;previous=-1;last=start
            while child.poll() is None:
                telemetry=gpu_state();progress=current/'progress.json'
                if progress.exists():
                    row=read(progress);state['progress']=row
                    if row['step']!=previous:previous=row['step'];last=time.monotonic()
                reason=requested or ((out/'STOP').exists() and 'owned STOP')
                if shutil.disk_usage(out).free<20*2**30:reason='disk floor'
                if time.monotonic()-start>limit:reason='bounded stage time'
                if time.monotonic()-last>1200:reason='no progress for20min'
                if reason and deadline is None:
                    current.mkdir(exist_ok=True);(current/'STOP').touch();deadline=time.monotonic()+180
                    state.update(status='STOPPING',stop_reason=reason)
                if deadline and time.monotonic()>deadline:raise TimeoutError('owned worker ignored STOP')
                state.update(gpu=telemetry,free_gib=shutil.disk_usage(out).free/2**30,elapsed_s=time.time()-state['started_at'])
                publish()
                with (out/'gpu.jsonl').open('a') as gpu:gpu.write(json.dumps(dict(time=time.time(),phase=name,gpus=telemetry))+'\n')
                time.sleep(10)
        record['exit_code']=child.returncode
        result=read(current/'result.json') if (current/'result.json').exists() else {}
        record['result']=result;publish();return result,child.returncode,record
    def config_file(batch):
        config=dict(base,batch_size=batch);path=out/('config-b%d.json'%batch);write(path,config);return config,path
    def converted(batch,world,new_rng):
        config,path=config_file(batch)
        packet=transition_local(source,config,a.previous_recipe,a.recipe,world,new_rng)
        packet['topology_transition'].update(parent_checkpoint=str(a.checkpoint.resolve()),git_commit=a.git_commit)
        target=out/'inputs'/('world%d-b%d-step%06d.pt'%(world,batch,parent_step))
        pending=target.with_suffix('.pt.part');torch.save(packet,pending);pending.replace(target)
        return path,target
    def training_command(name,config,checkpoint,world,extra):
        prefix=[sys.executable,'-u'] if world==1 else [sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=3']
        return prefix+[str(Path(__file__).with_name('train_pretrain.py')),'--config',str(config),
            '--manifest',str(a.recipe.resolve()),'--stats',str(a.recipe.parent/'stats.json'),
            '--output',str(out/name),'--workers','8' if world==3 else '4',
            '--git-commit',a.git_commit,'--resume',str(checkpoint)]+extra
    try:
        state['parent_audit']=audit(a.checkpoint,1,source['config']['batch_size'],parent_step,normalizer);publish()
        candidates=[]
        for batch in (96,128):
            released([4,5,6]);config,checkpoint=converted(batch,1,[]);name='capacity-b%d'%batch
            result,code,record=run(name,training_command(name,config,checkpoint,1,['--capacity-probe-updates','8']),
                                   dict(env,CUDA_VISIBLE_DEVICES='4'),600)
            if code:
                if 'out of memory' in (out/(name+'.log')).read_text().lower():continue
                raise RuntimeError('capacity probe failed: '+name)
            if result.get('status')!='COMPLETED':raise RuntimeError('capacity probe did not complete')
            row=dict(batch=batch,samples_per_s=result['steady_updates_per_s']*batch,
                peak_allocated_gib=result['peak_allocated_gib'],peak_reserved_gib=result['peak_reserved_gib'])
            record['capacity']=row
            if row['peak_reserved_gib']<=21:candidates.append(row)
            publish()
        if not candidates:raise RuntimeError('no larger batch passed memory headroom; preserve parent and inspect capacity logs')
        selected=max(candidates,key=lambda row:row['samples_per_s']);batch=selected['batch']
        state['capacity_selection']=dict(selected=selected,candidates=candidates);publish();released([4,5,6])
        # Generate new CUDA states in a short owned child with explicit device
        # visibility; it exits before training and leaves no retained contexts.
        rng_path=out/'inputs/new-rank-rng.pt'
        rng_code='''import random,sys,numpy as np,torch
states=[]
for rank in (1,2):
 seed=170*1000003+int(sys.argv[1])*97+rank
 states.append(dict(python=random.Random(seed).getstate(),numpy=np.random.RandomState(seed%(2**32)).get_state(),torch=torch.Generator().manual_seed(seed).get_state(),cuda=torch.Generator(device='cuda:'+str(rank)).manual_seed(seed).get_state().cpu()))
torch.save(states,sys.argv[2])
'''
        subprocess.run([sys.executable,'-c',rng_code,str(parent_step),str(rng_path)],env=env,check=True,timeout=60)
        rng=torch.load(rng_path,map_location='cpu',weights_only=False)
        config,checkpoint=converted(batch,3,rng)
        del source;gc.collect()
        # Parent CUDA generators can retain contexts on5/6; no model allocation.
        state.update(global_batch=batch*3,local_batch=batch,target_updates=10000,max_seconds=14400);publish()
        for name,updates in [('readiness',8),('strict-recovery',2),('training',None)]:
            extra=['--recovery-probe-updates',str(updates)] if updates else []
            result,code,record=run(name,training_command(name,config,checkpoint,3,extra),env,1500 if updates else 14700)
            if code:raise RuntimeError('three-GPU '+name+' failed; inspect log')
            if result['status']!='COMPLETED':state.update(status=result['status']);publish();return
            step=result['last_step'];checkpoint=out/name/('step%06d.pt'%step)
            record['checkpoint_audit']=audit(checkpoint,3,batch,step,normalizer);publish()
        state.update(status='COMPLETED');publish()
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));publish();raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':main()
