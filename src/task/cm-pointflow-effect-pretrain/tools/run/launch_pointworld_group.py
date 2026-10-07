#!/usr/bin/env python3
"""Launch/monitor three matched ref3 arms with immutable sources and one deadline."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parents[2]
REPO = TASK.parents[2]


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, value):
    part = path.with_suffix('.json.part')
    part.write_text(json.dumps(value, indent=2)+'\n')
    part.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stats', type=Path, required=True)
    p.add_argument('--config', type=Path, default=TASK/'configs/pointworld_small_wm24.json')
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root/'group_status.json'
    if state_path.exists(): raise FileExistsError('preserve previous launch record')
    config = json.loads(a.config.read_text())
    manifest = json.loads((a.data/'processed/manifest.json').read_text())
    if manifest['status']!='COMPLETED' or len(manifest['sequences'])!=627:
        raise ValueError('requires full verified 627-sequence corpus')
    if config['group_seconds']>86400: raise ValueError('24h group budget exceeded')
    import shutil
    if shutil.disk_usage(root).free < 10*1024**3: raise RuntimeError('need 10GiB free disk reserve')
    smi = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    available = [int(x.split(',')[0]) for x in smi.splitlines() if int(x.split(',')[1])<=512]
    if len(available)<3: raise RuntimeError('three free GPUs required; no process preemption')
    gpus = available[:3]
    files = [a.config.resolve(), a.stats.resolve(), TASK/'src/oakink_wm/pointworld.py',
             TASK/'src/oakink_wm/data.py', TASK/'src/oakink_wm/model.py',
             TASK/'tools/run/train_oakink2_pointworld.py', TASK/'tools/run/evaluate_oakink2_pointworld.py']
    files += [x for x in (REPO/'third_party/PointWorld/ptv3').rglob('*') if x.suffix in ('.py','.yaml')]
    identity = dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    dataset_manifest_sha256=digest(a.data/'processed/manifest.json'),
                    sources={str(x):digest(x) for x in files})
    deadline = time.time()+config['group_seconds']
    stop = [False]
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum,frame: stop.__setitem__(0,True))
    procs, logs = {}, []
    status = dict(status='LAUNCHING',pid=os.getpid(),identity=identity,gpus=gpus,
                  deadline=deadline,started_at=time.time(),updates=config['updates'])
    record(state_path,status)
    try:
        for gpu,arm in zip(gpus,('history','action','shuffle')):
            out = root/('train-'+arm)
            out.mkdir(exist_ok=False)
            log = (out/'console.log').open('w');logs.append(log)
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TMPDIR=str(REPO/'tmp'))
            procs[arm] = subprocess.Popen([sys.executable,'-u',str(TASK/'tools/run/train_oakink2_pointworld.py'),
                '--data',str(a.data.resolve()),'--config',str(a.config.resolve()),'--stats',str(a.stats.resolve()),
                '--output',str(out),'--arm',arm,'--deadline',str(deadline)],stdout=log,stderr=subprocess.STDOUT,
                env=env,start_new_session=True)
        status.update(status='TRAINING',processes={k:v.pid for k,v in procs.items()})
        record(state_path,status)
        stopped_at = None
        while any(v.poll() is None for v in procs.values()):
            failed = any(v.poll() not in (None,0) for v in procs.values())
            drift = any(digest(Path(x))!=h for x,h in identity['sources'].items())
            if failed or drift or stop[0] or time.time()>=deadline:
                if stopped_at is None:
                    stopped_at=time.time()
                    status['stop_reason']='failure' if failed else ('source_drift' if drift else ('user_stop' if stop[0] else 'deadline'))
                    for proc in procs.values():
                        if proc.poll() is None: os.kill(proc.pid,signal.SIGUSR1)
                elif time.time()-stopped_at>120:
                    for proc in procs.values():
                        if proc.poll() is None: os.killpg(proc.pid,signal.SIGTERM)
            status['progress']={}
            for arm in procs:
                try: status['progress'][arm]=json.loads((root/('train-'+arm)/'progress.json').read_text())
                except (FileNotFoundError,json.JSONDecodeError): pass
            record(state_path,status)
            time.sleep(30)
        codes={k:v.returncode for k,v in procs.items()}
        steps={k:json.loads((root/('train-'+k)/'progress.json').read_text()).get('step') for k in procs}
        complete=all(v==config['updates'] for v in steps.values()) and all(v==0 for v in codes.values())
        status.update(status='TEST_EVALUATION' if complete else 'STOPPED',exit_codes=codes,steps=steps,
                      matched_updates=len(set(steps.values()))==1)
        record(state_path,status)
        if complete and time.time()+120<deadline:
            for arm in procs:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpus[0]),TMPDIR=str(REPO/'tmp'))
                subprocess.run([sys.executable,str(TASK/'tools/run/evaluate_oakink2_pointworld.py'),
                    '--data',str(a.data.resolve()),'--checkpoint',str(root/('train-'+arm)/'final.pt'),
                    '--output',str(root/('train-'+arm)/'test_result.json')],env=env,check=True,
                    timeout=max(1,deadline-time.time()))
        status.update(status='COMPLETED' if complete else ('FAILED' if any(codes.values()) else 'BUDGET_STOP'),ended_at=time.time())
        record(state_path,status)
        return 0 if not any(codes.values()) else 1
    except BaseException as exc:
        status.update(status='FAILED',error=str(exc));record(state_path,status)
        for proc in procs.values():
            if proc.poll() is None: os.kill(proc.pid,signal.SIGUSR1)
        for proc in procs.values():
            if proc.poll() is None:
                try: proc.wait(timeout=120)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGTERM)
        raise
    finally:
        for log in logs: log.close()


if __name__=='__main__': raise SystemExit(main())
