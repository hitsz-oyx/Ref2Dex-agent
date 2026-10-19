#!/usr/bin/env python3
"""Run a prepared, bounded checkpoint panel with the existing native evaluator.

Preparation pins checkpoints and protocol in process.json; no policy training.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]


def write(path,value):
    part=path.with_suffix('.partial');part.write_text(json.dumps(value,indent=2)+'\n');part.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if ROOT/'outputs/wam-rl-policy-v2' not in out.parents:raise ValueError('Task-owned output required')
    status=json.loads((out/'process.json').read_text())
    if status['status']!='PREPARED':raise ValueError('Prepared fresh panel required')
    gpus=status['gpus']
    if (not 1<=len(gpus)<=4 or len(set(gpus))!=len(gpus) or any(g not in range(8) for g in gpus)
            or not 1<=len(status['checkpoints'])<=12 or not 1<=status['envs']<=256):
        raise ValueError('Bounded local checkpoint panel required')
    for gpu in status['gpus']:
        pids=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).split()
        for pid in pids:
            owner=subprocess.check_output(['ps','-o','user=','-p',pid],text=True).strip()
            if owner!='monica':raise RuntimeError('Unknown GPU occupant: '+pid)
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
        if used>2048:raise RuntimeError('Shared GPU memory budget')
    frozen=out/'runtime-code'
    shutil.copytree(TASK,frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    started=time.monotonic();active={};pending=list(status['checkpoints']);completed=[]
    status.update(status='RUNNING',pid=os.getpid(),launcher_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(out/'process.json',status)
    def launch(gpu,row):
        name=('panel-%d-'%row['panel'] if 'panel' in row else '')+'step-%d'%row['iteration']
        child_out=out/name
        command=[sys.executable,'-u',str(frozen/'tools/run/closed_loop_scheduled_shadow.py'),
            '--output',str(child_out),'--policy',status['policy'],'--object-manifest',row.get('object_manifest',status['object_manifest']),
            '--rl-checkpoint',row['checkpoint'],'--wam-checkpoint',status['wam_checkpoint'],
            '--wam-surface-manifest',status['wam_surface_manifest'],'--gpu',str(gpu),
            '--seed',str(status['seed']),'--envs',str(status['envs']),'--ppo-updates','0',
            '--official-ppo','--evaluate','--seconds',str(status['seconds_per_worker']),
            '--execution-protocol',status['execution_protocol'],'--goal-type',status['goal_type'],
            '--termination-policy',status['termination_policy'],
            '--checkpoint-migration-reason','Explicit evaluation of legacy unbound checkpoint using its original WAM12000/SE3/legacy execution protocol; adds outcome reporting only']
        env=dict(os.environ,REF2DEX_RUN_ROOT=str(ROOT),REF2DEX_BATCH_PIPELINE='all',REF2DEX_DEEP_OPT='all',
            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            LD_LIBRARY_PATH=str(Path(sys.executable).resolve().parents[1]/'lib'),
            TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/shadow-isaac-extensions'),TMPDIR=str(out/'tmp'))
        stream=(out/(name+'.log')).open('x')
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        row.update(gpu=gpu,pid=child.pid,command=command)
        return dict(child=child,stream=stream,row=row,output=child_out)
    try:
        while pending or active:
            for gpu in status['gpus']:
                if pending and gpu not in active:active[gpu]=launch(gpu,pending.pop(0))
            status['active']=[job['row'] for job in active.values()];status['completed']=completed
            write(out/'process.json',status)
            for gpu,job in list(active.items()):
                code=job['child'].poll()
                if code is None:continue
                job['stream'].close()
                if code or not (job['output']/'evaluation.json').is_file():raise RuntimeError('Checkpoint worker failed: '+str(job['row']['iteration']))
                report=json.loads((job['output']/'evaluation.json').read_text())
                if report['optimizer_steps'] or report['episodes']!=status['envs']:raise RuntimeError('Invalid evaluation result')
                completed.append(dict(iteration=report['iteration'],panel=job['row'].get('panel'),gpu=gpu,exit_code=code))
                del active[gpu]
            if time.monotonic()-started>status['total_seconds']:raise TimeoutError('Panel wall budget')
            if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('20GiB reserve')
            size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
            if size>status['artifact_limit_bytes']:raise RuntimeError('Artifact budget')
            usage=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,gpus)),'--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
            with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(elapsed_s=time.monotonic()-started,gpus=usage.splitlines()))+'\n')
            if active:time.sleep(10)
        status['status']='COMPLETED'
    except BaseException as error:
        status.update(status='FAILED',error=repr(error));raise
    finally:
        for job in active.values():
            child=job['child']
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            job['stream'].close()
        status.update(completed=completed,elapsed_s=time.monotonic()-started)
        write(out/'process.json',status)
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
