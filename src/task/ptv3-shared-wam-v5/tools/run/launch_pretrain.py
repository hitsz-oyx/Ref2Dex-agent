"""Own and monitor a bounded four-GPU v5 training process group."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import shutil

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v5.storage import read_json


def write(path,value):
    pending=path.with_suffix('.json.part');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path);path.chmod(0o664)


def gpu_rows():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
    return {int(s.split(',')[0]):dict(utilization=int(s.split(',')[1]),memory_mib=int(s.split(',')[2])) for s in raw.splitlines()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','manifest','stats','output','base-code-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--workers',type=int,default=8)
    p.add_argument('--git-commit',required=True);p.add_argument('--seed',type=int,default=170)
    p.add_argument('--loader-context',choices=('spawn','forkserver'),default='spawn')
    p.add_argument('--loader-timeout',type=int,default=120)
    p.add_argument('--recovery-probe-updates',type=int)
    p.add_argument('--resume',type=Path);a=p.parse_args()
    gpus=list(map(int,a.gpus.split(',')))
    if len(gpus)!=4 or len(set(gpus))!=4:raise ValueError('four distinct GPU IDs required')
    initial=gpu_rows()
    if any(g not in initial or initial[g]['memory_mib']>256 for g in gpus):raise ValueError('requested GPUs must be released by prior training')
    config=read_json(a.config);manifest=read_json(a.manifest)
    if not manifest.get('training_allowed') or manifest.get('engineering_subset') or manifest.get('pending_sequences',0):
        raise ValueError('complete admitted data required; never launch on arriving cache subset')
    if sorted(manifest['source_ids'])!=[0,1,2,4,5,6,8,9,10]:raise ValueError('all nine sources required')
    a.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,REF2DEX_V5_BASE_CODE_ROOT=str(a.base_code_root),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
        NCCL_DEBUG='WARN',NCCL_ASYNC_ERROR_HANDLING='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
        TMPDIR=str(a.output),TRITON_CACHE_DIR=str(a.output/'triton'),TORCH_EXTENSIONS_DIR=str(a.output/'extensions'))
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',
        str(Path(__file__).with_name('train_pretrain.py')),'--config',str(a.config),'--manifest',str(a.manifest),
        '--stats',str(a.stats),'--output',str(a.output),'--workers',str(a.workers),'--seed',str(a.seed),'--git-commit',a.git_commit]
    command+=['--loader-context',a.loader_context,'--loader-timeout',str(a.loader_timeout)]
    if a.resume:command+=['--resume',str(a.resume)]
    if a.recovery_probe_updates is not None:command+=['--recovery-probe-updates',str(a.recovery_probe_updates)]
    state=dict(status='STARTING',supervisor_pid=os.getpid(),command=command,git_commit=a.git_commit,
        work_version='v5.four-gpu-mixed-pretrain.v2',run_id=a.output.name,physical_gpus=gpus,
        global_batch=config['batch_size']*4,target_updates=config['updates'],
        max_seconds=config['max_seconds'],artifact_budget_bytes=config['artifact_budget_bytes'],
        dataset=str(a.manifest),seed=a.seed,started_at=time.time(),
        loader_context=a.loader_context,loader_timeout_s=a.loader_timeout,
        recovery_probe_updates=a.recovery_probe_updates,resume=str(a.resume) if a.resume else None)
    write(a.output/'process.json',state);requested=[]
    signal.signal(signal.SIGTERM,lambda *_:requested.append('supervisor SIGTERM'))
    signal.signal(signal.SIGINT,lambda *_:requested.append('supervisor SIGINT'))
    started=time.monotonic();child=None;deadline=None;last_step=-1;last_progress=started
    try:
        with (a.output/'train.log').open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['torchrun_pid']=child.pid
            while child.poll() is None:
                elapsed=time.monotonic()-started
                limit=min(config['max_seconds'],900) if a.recovery_probe_updates else config['max_seconds']
                stop=bool(requested) or (a.output/'STOP').exists() or elapsed>limit or shutil.disk_usage(a.output).free<20*2**30
                if stop and deadline is None:
                    (a.output/'STOP').touch();deadline=time.monotonic()+180
                    state.update(status='STOPPING',stop_reason=requested or 'STOP/deadline/free-space boundary')
                if deadline and time.monotonic()>deadline:raise TimeoutError('training did not finish controlled stop within180s')
                state.update(elapsed_s=elapsed,gpu=gpu_rows())
                progress=a.output/'progress.json'
                if progress.exists():
                    row=read_json(progress);state['progress']=row
                    if row['step']!=last_step:last_step=row['step'];last_progress=time.monotonic()
                    if deadline is None:state['status']='RUNNING'
                if time.monotonic()-last_progress>1800:raise TimeoutError('no training progress for30min')
                write(a.output/'process.json',state)
                with (a.output/'gpu.jsonl').open('a') as metrics:metrics.write(json.dumps(dict(elapsed_s=elapsed,gpus=state['gpu']))+'\n')
                time.sleep(15)
        result=read_json(a.output/'result.json') if (a.output/'result.json').exists() else {}
        state.update(status=result.get('status','FAILED') if child.returncode==0 else 'FAILED',exit_code=child.returncode,result=result)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state['total_elapsed_s']=time.monotonic()-started;write(a.output/'process.json',state)


if __name__=='__main__':main()
