#!/usr/bin/env python3
"""Supervise one warm-start action model on three assigned GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
REPO = TASK.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, value):
    part = path.with_suffix('.json.part')
    part.write_text(json.dumps(value, indent=2)+'\n')
    part.replace(path)


def gpu_processes(gpus):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    assigned = {uuid.strip() for row in rows.splitlines() for index,uuid in [row.split(',')] if int(index) in gpus}
    rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return [int(pid) for row in rows.splitlines() for uuid,pid in [row.split(',')] if uuid.strip() in assigned]


def belongs_to(pid, parent):
    """Only the torchrun subtree owned by this launcher may be signaled."""
    for _ in range(12):
        if pid==parent:
            return True
        try:
            fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
            pid = int(fields[1])
        except (FileNotFoundError, ProcessLookupError):
            return False
        if pid <= 1:
            return False
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data','stats','config','init-weights','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--gpus', default='0,1,2')
    p.add_argument('--deadline', type=float, required=True)
    a = p.parse_args()
    gpus = [int(x) for x in a.gpus.split(',')]
    config = json.loads(a.config.read_text())
    if len(gpus)!=3 or len(set(gpus))!=3 or config['accumulation']%3:
        raise ValueError('need three unique GPUs and accumulation divisible by3')
    if not time.time()<a.deadline<=time.time()+min(86400,config['group_seconds']):
        raise ValueError('deadline must be future and within authorized24h cap')
    if gpu_processes(gpus):
        raise RuntimeError('assigned GPU has another process; do not preempt it')
    if shutil.disk_usage(REPO).free < 10*1024**3:
        raise RuntimeError('need10GiB free disk reserve')
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__).resolve(), a.config.resolve(), a.stats.resolve(), a.data.resolve()/'processed/manifest.json',
             TASK/'src/oakink_wm/distributed.py', TASK/'src/oakink_wm/pointworld_performance.py',
             TASK/'src/oakink_wm/pointworld_temporal.py', TASK/'src/oakink_wm/pointworld.py',
             TASK/'src/oakink_wm/data.py', TASK/'src/oakink_wm/model.py',
             TASK/'tools/run/train_oakink2_pointworld_ddp.py', TASK/'tools/run/train_oakink2_pointworld_temporal.py',
             TASK/'tools/run/evaluate_oakink2_pointworld_temporal.py']
    files += [x for x in (REPO/'third_party/PointWorld/ptv3').rglob('*') if x.suffix in ('.py','.yaml')]
    sources = {str(x):digest(x) for x in files}
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,TMPDIR=str(REPO/'tmp'),
               TRITON_CACHE_DIR=str(REPO/'tmp/triton-pointworld-action-ddp-20261007'),
               OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=3',
               str(TASK/'tools/run/train_oakink2_pointworld_ddp.py'),'--data',str(a.data.resolve()),
               '--stats',str(a.stats.resolve()),'--config',str(a.config.resolve()),'--arm','action',
               '--init-weights',str(a.init_weights.resolve()),'--fused-hilbert','--deadline',str(a.deadline),
               '--output',str(root/'train-action')]
    status = dict(status='LAUNCHING',pid=os.getpid(),gpus=gpus,deadline=a.deadline,started_at=time.time(),
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  sources=sources,command=command)
    state_path = root/'group_status.json'
    record(state_path,status)
    stop = [False]
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda signum,frame:stop.__setitem__(0,True))
    def request_save(proc):
        manifest = root/'train-action/input_manifest.json'
        if manifest.exists():
            for rank in json.loads(manifest.read_text())['ranks']:
                pid = rank['pid']
                if belongs_to(pid,proc.pid):
                    try:
                        os.kill(pid,signal.SIGUSR1)
                    except ProcessLookupError:
                        pass
        else:
            # No worker checkpoint handler/identity is guaranteed at startup.
            os.killpg(proc.pid,signal.SIGTERM)
    with (root/'console.log').open('w') as log:
        proc = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        status.update(status='TRAINING',torchrun_pid=proc.pid)
        record(state_path,status)
        stopped_at = None
        try:
            while proc.poll() is None:
                foreign = [pid for pid in gpu_processes(gpus) if not belongs_to(pid,proc.pid)]
                drift = any(not Path(x).exists() or digest(Path(x))!=h for x,h in sources.items())
                requested = stop[0] or time.time()>=a.deadline or drift or foreign
                if requested and stopped_at is None:
                    status['stop_reason']='foreign_gpu_process' if foreign else ('source_drift' if drift else ('user_stop' if stop[0] else 'deadline'))
                    status['foreign_gpu_pids']=foreign
                    request_save(proc)
                    stopped_at=time.time()
                if stopped_at is not None and time.time()-stopped_at>180:
                    os.killpg(proc.pid,signal.SIGTERM)
                progress = root/'train-action/progress.json'
                if progress.exists():
                    status['progress']=json.loads(progress.read_text())
                record(state_path,status)
                time.sleep(30)
        except BaseException as exc:
            status.update(status='FAILED',error=str(exc))
            if proc.poll() is None:
                request_save(proc)
                try:
                    proc.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
            record(state_path,status)
            raise
        status.update(status='STOPPED' if status.get('stop_reason') else ('COMPLETED' if proc.returncode==0 else 'FAILED'),
                      exit_code=proc.returncode,ended_at=time.time())
        progress = root/'train-action/progress.json'
        if progress.exists():
            status['progress']=json.loads(progress.read_text())
        record(state_path,status)
    return proc.returncode


if __name__=='__main__':
    raise SystemExit(main())
