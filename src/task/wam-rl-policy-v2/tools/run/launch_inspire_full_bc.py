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
    p.add_argument('--current-run',type=Path,help='Reuse a completed full current BC and immutable physical cache')
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
        if a.current_run:
            import numpy as np
            import sys
            sys.path.insert(0,str(frozen/'src'))
            from wam_rl_policy_v2.inspire_shadow import CONTRACT,INPUT_DIM
            prior=a.current_run.resolve();cache=out/'cache';cache.mkdir()
            current_meta=json.loads((prior/'current-bc/manifest.json').read_text())
            if current_meta['status']!='COMPLETED' or current_meta['steps']!=50000:raise ValueError('completed full-data current BC required')
            meta=json.loads((prior/'cache/manifest.json').read_text())
            for name in ['stats.npz','dev-panel.npy','geometries','surface-manifest.json']+[s+'-'+k+'.npy' for s in ('train','dev') for k in ('x','y','tick','seq')]:
                (cache/name).symlink_to(prior/'cache'/name)
            meta.update(conditioning=CONTRACT,scheduled_status='PENDING',current_cache_source=str(prior/'cache'))
            (cache/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
            for s in ('train','dev'):
                n=meta['splits'][s]['commands']
                x=np.lib.format.open_memmap(cache/f'{s}-scheduled.npy',mode='w+',dtype=np.float32,shape=(n,INPUT_DIM));x[:,:312]=np.load(cache/f'{s}-x.npy',mmap_mode='r');x[:,312:]=0;x.flush()
                mask=np.lib.format.open_memmap(cache/f'{s}-include.npy',mode='w+',dtype=bool,shape=(n,));mask[:]=False;mask.flush()
            for rank in (0,1):np.save(cache/f'done-{rank}.npy',np.zeros(660,bool))
            current_parent=prior/'current-bc/best.pt';state['reused_current_bc']=str(current_parent)
            state['stage']='TWO_GPU_CONDITIONS';save()
            shard1=start('conditions','cache-gpu7',['--gpu','7','--rank','1'])
        else:
            prepare=start('prepare','prepare',[]);wait([prepare])
            state['stage']='CURRENT_BC_AND_GPU7_CACHE';save()
            current=start('current','current-bc',['--cache',str(out/'cache'),'--gpu','6'])
            shard1=start('conditions','cache-gpu7',['--gpu','7','--rank','1']);wait([current])
            current_parent=out/'current-bc/best.pt'
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
        scheduled=start('scheduled','scheduled-bc',['--cache',str(out/'cache'),'--gpu','6','--parent',str(current_parent)]);wait([scheduled])
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
