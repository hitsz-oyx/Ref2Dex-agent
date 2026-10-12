#!/usr/bin/env python3
"""Supervise remote DDP with a scoped short directory for multiprocessing sockets."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
REPO=TASK.parents[2]
HELPER=TASK.parent/'cm-pointflow-effect-pretrain/tools/run/launch_pointworld_action_ddp.py'
spec=importlib.util.spec_from_file_location('pointworld_launcher_helpers',HELPER)
helpers=importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','stats','init-pw','config','output','validation-indices'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpus',default='0,1,2,3')
    parser.add_argument('--runtime-tmp',type=Path,required=True)
    args=parser.parse_args(); root=args.output.resolve()
    runtime_tmp=args.runtime_tmp.resolve()
    if len(os.fsencode(str(runtime_tmp)))>75:
        raise ValueError('runtime tmp path exceeds AF_UNIX listener budget')
    if REPO.parent not in runtime_tmp.parents:
        raise ValueError('runtime tmp must stay inside the isolated NAS scope')
    runtime_tmp.mkdir(exist_ok=True)
    gpus=[int(x) for x in args.gpus.split(',')]
    config=json.loads(args.config.read_text())
    if len(gpus)!=4 or len(set(gpus))!=4 or config['world_size']!=4:
        raise ValueError('exactly four distinct GPU ranks required')
    if helpers.gpu_processes(gpus):
        raise RuntimeError('assigned GPU has an existing job')
    if (root/'group_status.json').exists() or (root/'train').exists():
        raise FileExistsError('fresh launcher/output required')
    root.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),HELPER,args.config,args.stats,args.data/'processed/manifest.json',
           args.validation_indices,args.validation_indices.parent/'panel.json',
           TASK/'tools/run/train_distributed.py',TASK/'tools/run/train_three_domain.py']
    files += list((TASK/'src/goal_point_wam').glob('*.py'))
    sources={str(p.resolve()):helpers.digest(p) for p in files}
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
             str(TASK/'tools/run/train_distributed.py')]
    for name in ('data','stats','init-pw','config','validation-indices'):
        command += ['--'+name,str(getattr(args,name.replace('-','_')).resolve())]
    command += ['--output',str(root/'train')]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpus,TMPDIR=str(runtime_tmp),
             TRITON_CACHE_DIR=str(REPO/'tmp/triton-goal-point-wam-ddp'),
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    started=time.time(); deadline=started+config['group_seconds']; stop=[False]
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    with (root/'console.log').open('w') as log:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status=dict(status='TRAINING',pid=os.getpid(),torchrun_pid=proc.pid,gpus=gpus,
                    started_at=started,deadline=deadline,command=command,sources=sources,
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        stopped_at=None; storage_check=0
        while proc.poll() is None:
            foreign=[pid for pid in helpers.gpu_processes(gpus) if not helpers.belongs_to(pid,proc.pid)]
            drift=any(not Path(p).exists() or helpers.digest(Path(p))!=h for p,h in sources.items())
            if time.time()-storage_check>=300:
                size=subprocess.check_output(['du','-sb',str(REPO/'outputs'),str(REPO/'tmp')],text=True)
                status['artifact_bytes']=sum(int(line.split()[0]) for line in size.splitlines())
                storage_check=time.time()
            reason=('foreign_gpu_process' if foreign else 'source_drift' if drift else
                    'storage_cap' if status.get('artifact_bytes',0)>=298_000_000_000 else
                    'user_stop' if stop[0] else 'deadline' if time.time()>=deadline else None)
            if reason and stopped_at is None:
                status.update(stop_reason=reason,foreign_gpu_pids=foreign)
                manifest=root/'train/input_manifest.json'
                if manifest.exists():
                    for worker in json.loads(manifest.read_text())['ranks']:
                        if helpers.belongs_to(worker['pid'],proc.pid):
                            try: os.kill(worker['pid'],signal.SIGUSR1)
                            except ProcessLookupError: pass
                else:
                    proc.terminate()
                stopped_at=time.time()
            if stopped_at is not None and time.time()-stopped_at>180:
                proc.terminate()
            progress=root/'train/progress.json'
            if progress.exists(): status['progress']=json.loads(progress.read_text())
            helpers.record(root/'group_status.json',status)
            time.sleep(30)
        status.update(status='STOPPED' if stopped_at else ('COMPLETED' if proc.returncode==0 else 'FAILED'),
                      exit_code=proc.returncode,ended_at=time.time())
        progress=root/'train/progress.json'
        if progress.exists(): status['progress']=json.loads(progress.read_text())
        helpers.record(root/'group_status.json',status)
    return proc.returncode


if __name__=='__main__':
    raise SystemExit(main())
