"""Freeze code, build full source inputs, run BC and two disjoint GPU cache shards."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[5]
PYTHON='/home2/wyy/miniconda3/envs/graspenv/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=28800);a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy-v2' not in out.parents or not 60<=a.seconds<=28800:raise ValueError('fresh bounded Task run required')
    if shutil.disk_usage(ROOT).free<24*2**30:raise RuntimeError('storage reserve')
    for gpu in (6,7):
        rows=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
        for row in rows.splitlines():
            pid,memory=map(lambda x:int(x.strip()),row.split(','));owner=subprocess.check_output(['ps','-o','user=','-p',str(pid)],text=True).strip()
            if owner!='monica' or memory>1024:raise RuntimeError('unexpected shared GPU occupant: '+row)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));out.mkdir(parents=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    frozen=out/'runtime-task';shutil.copytree(ROOT/'src/task/wam-rl-policy-v2',frozen,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    state=dict(status='RUNNING',stage='PREPARE_CURRENT',pid=os.getpid(),git_commit=commit,gpus=[6,7],run_id=out.name,
        max_seconds=a.seconds,max_artifact_bytes=2*1024**3,python=PYTHON,started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),children=[])
    path=out/'process.json';processes=[];handles=[];started=time.monotonic();sampled=-30
    def save():
        temp=path.with_suffix('.part');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(path)
    env=dict(os.environ,REF2DEX_RUN_ROOT=str(ROOT),REF2DEX_CODE_COMMIT=commit,PYTHONDONTWRITEBYTECODE='1')
    entry=frozen/'tools/run/inspire_full_bc.py'
    def start(stage,name,extra):
        args=[PYTHON,str(entry),stage,'--output',str(out/('cache' if stage in ('prepare','conditions') else name)),*extra]
        handle=(out/(name+'.log')).open('x');handles.append(handle)
        child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(child);state['children'].append(dict(name=name,pid=child.pid,command=args));save();return child
    def guard():
        nonlocal sampled
        elapsed=time.monotonic()-started
        if elapsed>a.seconds or shutil.disk_usage(out).free<20*2**30:raise RuntimeError('run time/storage floor')
        if (out/'STOP').exists():raise InterruptedError('owned STOP requested')
        if elapsed-sampled>=30:
            sampled=elapsed;size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
            if size>state['max_artifact_bytes']:raise RuntimeError('2GiB output limit')
            gpu=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
            with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(elapsed_s=elapsed,gpus=gpu,artifact_bytes=size,stage=state['stage']))+'\n')
            print(json.dumps(dict(stage=state['stage'],elapsed_s=elapsed,gpus=gpu)),flush=True)
    def wait(children):
        while True:
            guard();codes=[p.poll() for p in children]
            if any(c not in (None,0) for c in codes):raise RuntimeError('stage exited '+str(codes))
            if all(c==0 for c in codes):return
            time.sleep(2)
    def stop(*_):raise InterruptedError('supervisor stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        prepare=start('prepare','prepare',[]);wait([prepare])
        state['stage']='CURRENT_BC_AND_GPU7_CACHE';save()
        current=start('current','current-bc',['--cache',str(out/'cache'),'--gpu','6'])
        shard1=start('conditions','cache-gpu7',['--gpu','7','--rank','1']);wait([current])
        if shard1.poll() not in (None,0):raise RuntimeError('GPU7 condition worker failed')
        state['stage']='TWO_GPU_CONDITIONS';save()
        shard0=start('conditions','cache-gpu6',['--gpu','6','--rank','0']);wait([shard0,shard1])
        import numpy as np
        done=np.load(out/'cache/done-0.npy')|np.load(out/'cache/done-1.npy')
        if not done.all():raise RuntimeError('condition shards did not cover all660 sequences')
        path_meta=out/'cache/manifest.json';meta=json.loads(path_meta.read_text());meta['scheduled_status']='COMPLETED'
        meta['scheduled_counts']={s:int(np.load(out/f'cache/{s}-include.npy').sum()) for s in ('train','dev')}
        path_meta.write_text(json.dumps(meta,indent=2)+'\n')
        state['stage']='SCHEDULED_BC';save()
        scheduled=start('scheduled','scheduled-bc',['--cache',str(out/'cache'),'--gpu','6','--parent',str(out/'current-bc/best.pt')]);wait([scheduled])
        state.update(status='COMPLETED',stage='READY_FOR_SHADOW_PPO')
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));raise
    finally:
        for child in processes:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in processes:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for handle in handles:handle.close()
        state.update(elapsed_s=time.monotonic()-started,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'));save()


if __name__=='__main__':main()
