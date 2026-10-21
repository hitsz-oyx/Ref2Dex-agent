"""Approved wait-for-release queue: same three-rank run with geometry loss disabled."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import shutil
import torch

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from launch_local_training import read,write,gpu_state,released,alive
from launch_local_scaled import audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--approved-wait-for-release',action='store_true',required=True)
    for k in ('geometry-run','source','recipe','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--git-commit',required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
    torch.set_num_threads(2);old=read(a.geometry_run/'process.json');old_pids=[old['supervisor_pid'],old['stages'][-1]['pid']]
    if old['run_id']!='geometry-consistency-20261021-r1' or old['physical_gpus']!=[4,5,6]:raise ValueError('unexpected active run')
    original=read(a.geometry_run/'config-b96.json');config=dict(original,geometry_loss_disabled=True)
    difference={k:[original.get(k),config.get(k)] for k in config if config.get(k)!=original.get(k)}
    if difference!={'geometry_loss_disabled':[None,True]}:raise ValueError('only the geometry-loss ablation may differ')
    write(out/'config.json',config)
    state=dict(status='QUEUED',phase='wait-natural-release',work_version='v6.matched-no-geometry-control.v1',
        experiment_id='P-20261021-wam-v6-geometry-ab',run_id=out.name,supervisor_pid=os.getpid(),
        git_commit=a.git_commit,started_at=time.time(),physical_gpus=[4,5,6],global_batch=288,
        source=str(a.source.resolve()),source_step=4371,config_diff=difference,stages=[],
        resource_plan='explicit approved wait for active run to finish; no external signals/STOP files',
        checkpoint_policy='preserve every new checkpoint;32GiB/stage ceiling,20GiB free floor')
    def publish():write(out/'process.json',state)
    publish();requested=[];child=None
    signal.signal(signal.SIGTERM,lambda *_:requested.append('SIGTERM'))
    signal.signal(signal.SIGINT,lambda *_:requested.append('SIGINT'))
    try:
        deadline=time.monotonic()+21600
        while True:
            current=read(a.geometry_run/'process.json');telemetry=gpu_state()
            if requested or (out/'STOP').exists():state.update(status='CANCELLED');publish();return
            if time.monotonic()>deadline:raise TimeoutError('six-hour queue deadline')
            if shutil.disk_usage(out).free<60*2**30:raise ValueError('60GiB control peak reservation not available')
            state.update(waiting_for_step=current.get('progress',{}).get('step'),gpu=telemetry,free_gib=shutil.disk_usage(out).free/2**30);publish()
            if current['status']=='FAILED':raise ValueError('geometry run failed; user review required, no automatic replacement')
            if current['status'] in ('COMPLETED','STOPPED') and not any(alive(pid) for pid in old_pids):
                if all(telemetry[g]['memory_mib']<=256 for g in [4,5,6]):break
                state['waiting_reason']='another process has claimed a requested GPU';publish()
            time.sleep(60)
        released([4,5,6]);saved=torch.load(a.source,map_location='cpu',weights_only=False)
        if saved['step']!=4371 or saved['config'].get('geometry_consistency'):raise ValueError('exact baseline4371 required')
        normalizer={k:v.clone() for k,v in saved['model'].items() if k.startswith('normalizer.')}
        state['source_audit']=audit(a.source,3,96,4371,normalizer);del saved;gc.collect();publish()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
            PYTHONDONTWRITEBYTECODE='1',NCCL_P2P_DISABLE='1',NCCL_DEBUG='WARN',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
            NCCL_ASYNC_ERROR_HANDLING='1',TMPDIR=str(out),TRITON_CACHE_DIR=str(out/'triton'),TORCH_EXTENSIONS_DIR=str(out/'extensions'))
        checkpoint=None
        for name,extra in [('readiness',['--steps','8']),('strict-recovery',['--recovery-probe-updates','2']),('training',[])]:
            released([4,5,6]);stage=out/name
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=3',
                str(Path(__file__).with_name('train_pretrain.py')),'--config',str(out/'config.json'),
                '--manifest',str(a.recipe.resolve()),'--stats',str(a.recipe.parent/'stats.json'),'--output',str(stage),
                '--workers','8','--git-commit',a.git_commit,'--preserve-all-checkpoints','--checkpoint-budget-gib','32']
            command+=['--resume',str(checkpoint)] if checkpoint else ['--initialize-geometry-from',str(a.source.resolve())]
            command+=extra;record=dict(name=name,command=command,started_at=time.time());state['stages'].append(record)
            state.update(status='RUNNING',phase=name);state.pop('progress',None);publish()
            with (out/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                record['pid']=child.pid;publish();last=time.monotonic();previous=-1;stop_deadline=None
                while child.poll() is None:
                    progress=stage/'progress.json'
                    if progress.exists():
                        row=read(progress);state['progress']=row
                        if row['step']!=previous:previous=row['step'];last=time.monotonic()
                    reason=requested or ((out/'STOP').exists() and 'owned STOP')
                    if shutil.disk_usage(out).free<20*2**30:reason='disk floor'
                    if time.monotonic()-last>1200:reason='no progress20min'
                    if time.time()-record['started_at']>(15000 if name=='training' else 1800):reason='stage deadline'
                    if reason and stop_deadline is None:
                        stage.mkdir(exist_ok=True);(stage/'STOP').touch();stop_deadline=time.monotonic()+180
                        state.update(status='STOPPING',stop_reason=reason)
                    if stop_deadline and time.monotonic()>stop_deadline:raise TimeoutError('owned control child ignored STOP')
                    state.update(gpu=gpu_state(),free_gib=shutil.disk_usage(out).free/2**30);publish();time.sleep(15)
            record['exit_code']=child.returncode
            if child.returncode:raise RuntimeError('control stage failed '+name)
            result=read(stage/'result.json');record['result']=result;step=result['last_step'];checkpoint=stage/('step%06d.pt'%step)
            record['checkpoint_audit']=audit(checkpoint,3,96,step,normalizer);publish()
            if result['status']!='COMPLETED':state.update(status=result['status']);publish();return
        state.update(status='COMPLETED');publish()
    except BaseException as exc:state.update(status='FAILED',error=repr(exc));publish();raise
    finally:
        # Signals are restricted to this queue's own newly launched process group.
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':main()
