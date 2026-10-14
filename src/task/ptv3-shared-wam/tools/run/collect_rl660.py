"""Supervise a bounded two-GPU full-corpus collection; freeze only after 660 receipts."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
RUNTIME = '/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python'


def write(path,value):
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experiment-card',type=Path,required=True)
    p.add_argument('--gpus',default='4,5')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seed',type=int,default=120)
    p.add_argument('--seconds',type=int,default=10800)
    p.add_argument('--detach',action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    gpus = a.gpus.split(',')
    if (out.exists() or ROOT/'outputs/ptv3-shared-wam' not in out.parents
            or len(gpus) != 2 or len(set(gpus)) != 2 or not 1<=a.batch_size<=32
            or not a.experiment_card.is_file()):
        raise ValueError('fresh owned output, two GPUs and existing experiment card required')
    scratch = ROOT/'tmp'/out.name
    scratch.mkdir(parents=True,exist_ok=True)
    if a.detach:
        command = [sys.executable,str(Path(__file__).resolve()),'--output',str(out),
            '--experiment-card',str(a.experiment_card.resolve()),'--gpus',a.gpus,
            '--batch-size',str(a.batch_size),'--seed',str(a.seed),'--seconds',str(a.seconds)]
        with (scratch/'supervisor.log').open('x') as log:
            process = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,TMPDIR=str(scratch),PYTHONDONTWRITEBYTECODE='1'))
        write(scratch/'launch.json',dict(pid=process.pid,command=command))
        print(json.dumps(dict(supervisor_pid=process.pid,scratch=str(scratch))),flush=True)
        return
    for gpu in gpus:
        if subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('assigned GPU occupied: '+gpu)
    source = ROOT/'data/processed_data/inspire_rl_object_dexplore'
    names = sorted(p.parent.name for p in source.glob('*/interaction_hand_inspire.pt'))
    if len(names)!=660:
        raise ValueError('frozen 660-member source required')
    out.mkdir(parents=True)
    manifest = dict(schema='ref2dex.rl660-corpus.v1',status='RUNNING',training_allowed=False,
        engineering_only=False,num_sequences=660,fps=30,records=[],physical_gpus=gpus,
        experiment_card=str(a.experiment_card.resolve()),seed=a.seed,
        source_membership=str(source.resolve()),supervisor_pid=os.getpid(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        split_policy='complete sequence; stratified by object, 10% dev with at least one per object')
    write(out/'manifest.json',manifest)
    batches = [names[i:i+a.batch_size] for i in range(0,len(names),a.batch_size)]
    pending = list(enumerate(batches))
    active,completed = {},[]
    started = time.monotonic()
    storage_checked = 0.
    stopped = []
    signal.signal(signal.SIGTERM,lambda *_:stopped.append(True))
    signal.signal(signal.SIGINT,lambda *_:stopped.append(True))
    try:
        while pending or active:
            if stopped or time.monotonic()-started>a.seconds:
                raise TimeoutError('requested stop or collection deadline')
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                index,batch = pending.pop(0)
                selection = scratch/('selection%03d.json'%index)
                write(selection,batch)
                folder = out/'batches'/('batch%03d'%index)
                folder.parent.mkdir(exist_ok=True)
                command = [RUNTIME,str(TASK/'tools/run/collect_rl660_batch.py'),
                    '--selection',str(selection),'--output',str(folder),'--gpu',gpu,
                    '--seed',str(a.seed+index),'--seconds','900']
                log = (scratch/('batch%03d.log'%index)).open('x')
                process = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,TMPDIR=str(scratch),PYTHONDONTWRITEBYTECODE='1'))
                active[gpu] = (process,log,folder,index)
            for gpu,(process,log,folder,index) in list(active.items()):
                if process.poll() is None:
                    continue
                log.close()
                if process.returncode:
                    raise RuntimeError('collection batch%d failed; see supervisor scratch'%index)
                receipt = json.loads((folder/'manifest.json').read_text())
                if receipt['status']!='COMPLETED':
                    raise ValueError('incomplete batch receipt')
                controller = receipt['native_controller']
                if manifest.get('native_controller',controller)!=controller:
                    raise ValueError('native controller changes across sequences')
                manifest['native_controller'] = controller
                for record in receipt['records']:
                    record['path'] = str((folder/record['path']).relative_to(out))
                    completed.append(record)
                del active[gpu]
            elapsed = time.monotonic()-started
            row = dict(status='RUNNING',completed_sequences=len(completed),target=660,
                elapsed_s=elapsed,eta_s=elapsed/max(1,len(completed))*(660-len(completed)),
                active_pids=[v[0].pid for v in active.values()],
                gpu=subprocess.check_output(['nvidia-smi','-i',a.gpus,
                    '--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())
            write(out/'progress.json',row)
            print(json.dumps(row),flush=True)
            if len(completed)!=len({r['sequence'] for r in completed}):
                raise ValueError('duplicate collected sequence')
            size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
            if size>4_000_000_000:
                raise RuntimeError('bounded corpus storage exceeded')
            if elapsed-storage_checked>=120 or storage_checked==0:
                usage=subprocess.check_output(['du','-sb',str(ROOT/'outputs'),str(ROOT/'tmp')],text=True)
                if sum(int(line.split()[0]) for line in usage.splitlines())>=400_000_000_000:
                    raise RuntimeError('shared 400GB artifact budget exceeded')
                storage_checked=elapsed
            if pending or active:
                time.sleep(15)
        if sorted(r['sequence'] for r in completed)!=names:
            raise ValueError('incomplete 660-member execution corpus')
        # Stable per-object sequence split independent of GPU completion order.
        import random
        rng = random.Random(a.seed+1)
        objects = sorted({r['object'] for r in completed})
        for obj in objects:
            group=sorted([r for r in completed if r['object']==obj],key=lambda r:r['sequence'])
            rng.shuffle(group)
            count=max(1,round(len(group)*.1))
            for index,record in enumerate(group):
                record['split']='val' if index<count else 'train'
        manifest.update(status='COMPLETED',training_allowed=True,
            records=sorted(completed,key=lambda r:r['sequence']),elapsed_s=time.monotonic()-started)
        import numpy as np
        manifest['max_reference_error_m'] = .1
        for record in manifest['records']:
            with np.load(out/record['path']/'trajectory.npz') as data:
                error = np.linalg.norm(data['obj'][:,:3,3]-data['reference_obj'][:,:3,3],axis=-1)
            count = len(error)-24
            peak = np.maximum.reduce([error[i:i+count] for i in range(25)])
            ticks = np.flatnonzero(np.isfinite(peak) & (peak<=.1))
            record.update(raw_windows=count,admitted_windows=len(ticks),
                admitted_ticks=ticks.tolist(),excluded_windows=count-len(ticks))
        manifest['admitted_windows'] = sum(r['admitted_windows'] for r in manifest['records'])
        write(out/'manifest.json',manifest)
        write(out/'progress.json',dict(status='COMPLETED',completed_sequences=660,elapsed_s=manifest['elapsed_s']))
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error),records=completed)
        write(out/'manifest.json',manifest)
        raise
    finally:
        for process,log,_,_ in active.values():
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            log.close()


if __name__=='__main__':
    main()
