"""Own only the authorized two-GPU expanded Shadow training process group."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',type=Path,required=True);p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',default='4,5')
    p.add_argument('--updates',type=int,default=20000);p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--detach',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();out=a.output.resolve();gpus=a.gpus.split(',')
    if len(gpus)!=2 or len(set(gpus))!=2 or any(not x.isdigit() for x in gpus):raise ValueError('Two distinct GPUs')
    if out.exists() or ROOT/'outputs/shadow-finetune' not in out.parents:raise ValueError('Fresh Shadow output')
    scratch=ROOT/'tmp'/out.name;scratch.mkdir(parents=True,exist_ok=True)
    if a.detach:
        cmd=[sys.executable,str(Path(__file__).resolve())]+[x for x in sys.argv[1:] if x!='--detach']
        with (scratch/'supervisor.log').open('x') as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        receipt=dict(supervisor_pid=child.pid,output=str(out),gpus=gpus,command=cmd)
        (scratch/'detach.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return
    def gpu_pids():
        s=subprocess.check_output(['nvidia-smi','-i',','.join(gpus),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
        return {int(x) for x in s.split() if x.isdigit()}
    if gpu_pids():raise RuntimeError('Assigned GPU occupied')
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        str(TASK/'tools/run/train_expanded.py'),'--resume',str(a.resume.resolve()),'--controls',str(a.controls.resolve()),
        '--output',str(out),'--updates',str(a.updates),'--batch-size',str(a.batch_size),'--workers',str(a.workers)]
    if a.smoke:cmd.append('--smoke')
    status=dict(status='STARTING',supervisor_pid=os.getpid(),command=cmd,gpus=gpus,
        authorization='User reallocates former shared_tau GPUs to expanded Shadow and allows idle GPUs for evaluation')
    def write():
        part=scratch/'process.json.part';part.write_text(json.dumps(status,indent=2)+'\n');part.replace(scratch/'process.json')
        if out.exists():
            part=out/'process.json.part';part.write_text(json.dumps(status,indent=2)+'\n');part.replace(out/'process.json')
    stop=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda n,f:stop.append(n))
    started=time.monotonic();child=None;last_step=-1;last_progress=started;write()
    try:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(gpus),REF2DEX_RUN_ROOT=str(ROOT),TMPDIR=str(scratch),
            TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/shared-wam-v2-extensions'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        with (scratch/'train.log').open('x') as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
        status.update(status='RUNNING',torchrun_pid=child.pid);write()
        while child.poll() is None:
            if stop:raise RuntimeError('User signal '+str(stop))
            if time.monotonic()-started>43800:raise TimeoutError('12h training+startup deadline')
            if shutil.disk_usage(ROOT).free<20*2**30:raise RuntimeError('20GiB free floor')
            for pid in gpu_pids():
                try:group=os.getpgid(pid)
                except ProcessLookupError:continue
                if group!=child.pid:raise RuntimeError('Foreign selected-GPU process')
            path=out/'progress.json'
            if path.exists():
                progress=json.loads(path.read_text());status['progress']=progress
                if progress['step']!=last_step:last_step=progress['step'];last_progress=time.monotonic()
            if time.monotonic()-last_progress>1800:raise RuntimeError('No progress for30min')
            status['gpu']=subprocess.check_output(['nvidia-smi','-i',','.join(gpus),
                '--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
            write();time.sleep(10)
        status.update(status='COMPLETED' if child.returncode==0 else 'FAILED',exit_code=child.returncode)
    except BaseException as error:
        status.update(status='STOPPED' if stop else 'FAILED',error=repr(error))
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        status['elapsed_s']=time.monotonic()-started;write()
    if status['status']!='COMPLETED':raise SystemExit(1)


if __name__=='__main__':main()
