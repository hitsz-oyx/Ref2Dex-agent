"""Bounded handoff: wait for the frozen 660 corpus, audit it, then launch robot FT."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0,str(TASK/'tools/run'))
from collect_rl660 import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source','output','config','experiment-card','init-shared'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--gpus',required=True)
    parser.add_argument('--wait-seconds',type=int,default=10800)
    parser.add_argument('--detach',action='store_true')
    args = parser.parse_args()
    args.source,args.output = args.source.resolve(),args.output.resolve()
    if (ROOT/'outputs/ptv3-shared-wam' not in args.source.parents
            or ROOT/'outputs/ptv3-shared-wam' not in args.output.parents
            or args.output.exists() or not 1<=args.wait_seconds<=10800):
        raise ValueError('owned source, fresh output and bounded collection wait required')
    scratch = ROOT/'tmp'/(args.output.name+'-handoff')
    scratch.mkdir(parents=True,exist_ok=True)
    state_path = scratch/'status.json'
    if args.detach:
        command = [sys.executable,str(Path(__file__).resolve())]
        for key in ('source','output','config','experiment_card','init_shared','gpus','wait_seconds'):
            command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
        with (scratch/'supervisor.log').open('x') as log:
            process = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,TMPDIR=str(scratch),PYTHONDONTWRITEBYTECODE='1',
                    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        write(scratch/'launch.json',dict(pid=process.pid,command=command))
        print(json.dumps(dict(pid=process.pid,status_path=str(state_path))),flush=True)
        return
    started = time.monotonic()
    state = dict(status='WAITING_FOR_COLLECTION',source=str(args.source),output=str(args.output),pid=os.getpid())
    write(state_path,state)
    try:
        while True:
            manifest = json.loads((args.source/'manifest.json').read_text())
            if manifest['status'] == 'FAILED':
                raise RuntimeError('collection failed: '+manifest.get('error','unknown'))
            if manifest['status'] == 'COMPLETED':
                break
            if time.monotonic()-started>args.wait_seconds:
                raise TimeoutError('bounded collection handoff wait expired')
            time.sleep(30)
        if manifest['num_sequences'] != 660 or manifest.get('engineering_only'):
            raise ValueError('full scientific 660-member corpus required')
        state['status'] = 'AUDITING_CORPUS'
        write(state_path,state)
        sys.path.insert(0,str(TASK/'src'))
        from ptv3_shared_wam.rl_robot import RLRobotWindows
        # File/label audit uses CPU; no neural model computation here.
        quality = {}
        for split in ('train','val'):
            dataset = RLRobotWindows(args.source,split)
            for index in range(len(dataset.records)):
                dataset.sequence(index)
            quality[split] = dict(sequences=len(dataset.records),windows=len(dataset),
                contributing_sequences=sum(bool(len(t)) for t in dataset.ticks_by_sequence),
                zero_window_sequences=[r['sequence'] for r,t in zip(dataset.records,dataset.ticks_by_sequence) if not len(t)])
            dataset.sequence.cache_clear()
        state['quality'] = quality
        write(scratch/'quality.json',quality)
        command = [sys.executable,str(TASK/'tools/run/launch_pretrain.py'),'--detach',
            '--stage','robot','--gpus',args.gpus,'--config',str(args.config.resolve()),
            '--experiment-card',str(args.experiment_card.resolve()),'--source',str(args.source),
            '--init-shared',str(args.init_shared.resolve()),'--output',str(args.output),
            '--loader-workers','4','--loader-prefetch','2','--sequence-cache','128','--pin-memory']
        launch = subprocess.check_output(command,cwd=ROOT,text=True)
        state.update(status='TRAINING_LAUNCHED',launch=json.loads(launch))
        write(state_path,state)
        observe_started = time.monotonic()
        while time.monotonic()-observe_started<1800:
            process_path = args.output/'process.json'
            if process_path.is_file():
                process = json.loads(process_path.read_text())
                if process['status'] in ('FAILED','COMPLETED'):
                    state.update(status=process['status'],training_process=process)
                    write(state_path,state)
                    return
            progress_path = args.output/'progress.json'
            if progress_path.is_file():
                progress = json.loads(progress_path.read_text())
                state.update(progress=progress,gpu=subprocess.check_output(['nvidia-smi','-i',args.gpus,
                    '--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())
                if progress['step']>=40:
                    state['status']='TRAINING_OBSERVED'
                    write(state_path,state)
                    return
                write(state_path,state)
            time.sleep(30)
        state['status']='TRAINING_LAUNCHED_OBSERVATION_TIMEOUT'
        write(state_path,state)
    except BaseException as error:
        state.update(status='FAILED',error=repr(error))
        write(state_path,state)
        raise


if __name__=='__main__':
    main()
