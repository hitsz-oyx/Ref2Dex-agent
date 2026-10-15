"""Own and supervise only this frozen two-GPU training process group."""
import argparse
import json
import os
from pathlib import Path
import shutil
import resource
import signal
import subprocess
import sys
import time
TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]


def write_json(path,value):
    temp=path.with_suffix('.json.part');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def gpu_rows():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(r.split(',')[0]):dict(memory_mib=float(r.split(',')[1]),utilization=float(r.split(',')[2])) for r in text.splitlines()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',default='1,3')
    p.add_argument('--loader-workers',type=int,default=2)
    p.add_argument('--detach',action='store_true')
    p.add_argument('--stop-after',type=int)
    p.add_argument('--resume',type=Path)
    a=p.parse_args()
    artifacts,out=a.artifact_root.resolve(),a.output.resolve()
    if artifacts/'outputs'/TASK.name not in out.parents or out.exists():raise ValueError('fresh v3 output required')
    gpus=list(map(int,a.gpus.split(',')))
    if len(set(gpus))!=len(gpus) or not 1<=len(gpus)<=4:raise ValueError('one to four GPUs required')
    config=json.loads(a.config.read_text())
    if config.get('expected_world_size',len(gpus))!=len(gpus):
        raise ValueError('assigned GPUs differ from explicit execution profile')
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    target=65536 if hard==resource.RLIM_INFINITY else min(65536,hard)
    if soft<target:resource.setrlimit(resource.RLIMIT_NOFILE,(target,hard))
    scratch=artifacts/'tmp'/out.name;scratch.mkdir(parents=True,exist_ok=True)
    if a.detach:
        command=[sys.executable,str(Path(__file__).resolve())]+[v for v in sys.argv[1:] if v!='--detach']
        with (scratch/'supervisor.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(scratch)))
        receipt=dict(supervisor_pid=process.pid,output=str(out),scratch=str(scratch),code_root=str(ROOT),command=command)
        write_json(scratch/'launch.json',receipt);print(json.dumps(receipt),flush=True);return
    sys.path.insert(0,str(TASK/'src'))
    from ptv3_shared_wam_v3.checkpoint import directory_bytes
    started=time.monotonic(); process=None; requested=[]
    signal.signal(signal.SIGTERM,lambda n,f:requested.append('supervisor SIGTERM'))
    signal.signal(signal.SIGINT,lambda n,f:requested.append('supervisor SIGINT'))
    status=dict(status='WAITING_FOR_GPU',supervisor_pid=os.getpid(),physical_gpus=gpus,code_root=str(ROOT),
        target_updates=config['updates'],global_batch=config['batch_size']*len(gpus),output=str(out))
    try:
        while True:
            rows=gpu_rows();active={i for i,r in rows.items() if r['memory_mib']>256}
            if len(active|set(gpus))<=6 and not active.intersection(gpus):
                occupied=any(subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() for g in gpus)
                if not occupied:break
            status.update(elapsed_s=time.monotonic()-started,gpu=rows)
            write_json(scratch/'process.json',status)
            if requested or (scratch/'STOP').exists():raise RuntimeError('own queue stopped')
            if time.monotonic()-started>1800:raise TimeoutError('GPU queue deadline30min')
            time.sleep(10)
        if shutil.disk_usage(artifacts).free<20*2**30:raise RuntimeError('free disk below20GiB')
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,TMPDIR=str(scratch),
            TRITON_CACHE_DIR=str(scratch/'triton'),TORCH_EXTENSIONS_DIR=str(scratch/'extensions'),
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            NCCL_DEBUG='WARN',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',PYTHONDONTWRITEBYTECODE='1')
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
            '--nproc_per_node='+str(len(gpus)),str(TASK/'tools/run/train_pretrain.py'),
            '--config',str(a.config.resolve()),'--recipe',str(a.recipe.resolve()),'--output',str(out),
            '--artifact-root',str(artifacts),'--loader-workers',str(a.loader_workers)]
        if a.resume:command+=['--resume',str(a.resume.resolve())]
        if a.stop_after:command+=['--stop-after',str(a.stop_after)]
        status.update(status='STARTING',command=command)
        last_progress=time.monotonic();last_step=-1;checked=0.;launched=time.monotonic()
        with (scratch/'train.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=environment)
            status['torchrun_pid']=process.pid
            while process.poll() is None:
                elapsed=time.monotonic()-launched;rows=gpu_rows()
                active={i for i,r in rows.items() if r['memory_mib']>256}
                if len(active|set(gpus))>6:raise RuntimeError('aggregate6-GPU budget exceeded; stopping own group')
                if requested or (out/'STOP').exists() or (scratch/'STOP').exists():raise RuntimeError('own STOP requested')
                if elapsed>config['max_seconds']:raise TimeoutError('48h training deadline')
                if shutil.disk_usage(artifacts).free<20*2**30:raise RuntimeError('free disk below20GiB')
                if elapsed-checked>=60 or checked==0:
                    owned=directory_bytes(scratch,*([out] if out.exists() else []))
                    if owned>config['artifact_budget_bytes']:raise RuntimeError('owned artifact budget exceeded')
                    status['artifact_bytes']=owned;checked=elapsed
                progress=out/'progress.json'
                if progress.exists():
                    row=json.loads(progress.read_text());status['progress']=row
                    if row['step']!=last_step:last_step=row['step'];last_progress=time.monotonic()
                if time.monotonic()-last_progress>1800:raise TimeoutError('no training progress30min')
                status.update(status='RUNNING' if last_step>=0 else 'STARTING',elapsed_s=elapsed,gpu=rows)
                with (scratch/'gpu.jsonl').open('a') as f:f.write(json.dumps(dict(elapsed_s=elapsed,gpus=rows))+'\n')
                write_json(scratch/'process.json',status)
                if out.exists():
                    write_json(out/'process.json',status)
                    for name,target in [('launcher.log',scratch/'train.log'),('gpu.jsonl',scratch/'gpu.jsonl')]:
                        if not (out/name).exists():(out/name).symlink_to(target)
                time.sleep(10)
            status.update(status='COMPLETED' if process.returncode==0 else 'FAILED',exit_code=process.returncode)
    except BaseException as error:
        status.update(status='FAILED',error=repr(error))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            status['exit_code']=process.returncode
        status['total_elapsed_s']=time.monotonic()-started
        write_json(scratch/'process.json',status)
        if out.exists():write_json(out/'process.json',status)
    if status['status']!='COMPLETED':raise SystemExit(1)


if __name__=='__main__':main()
