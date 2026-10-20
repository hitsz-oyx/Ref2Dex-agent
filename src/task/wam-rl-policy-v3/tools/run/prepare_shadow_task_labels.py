"""Replay task10 independently of neural WAM inference so Current310 can start."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()
sys.path.insert(0,str(TASK/'src'))
_DATASET=None


def initialize(source,records,geometries):
    global _DATASET
    from wam_rl_policy_v3.scheduled_current_cache import RecordInputs
    _DATASET=RecordInputs(source,records,geometries,task_goal=True)


def read(index):
    result=_DATASET[index]
    return index,result['task']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--dense',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--seconds',type=int,default=7200)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents or not 1<=a.workers<=16:
        raise ValueError('fresh bounded task-label output required')
    out.mkdir(parents=True)
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    import subprocess
    import numpy as np
    from wam_rl_policy_v3.wam_v4_runtime import load_geometry
    from wam_rl_policy_v3.shadow_task_contract import conditioning
    from wam_rl_policy_v3.executor_contract import TASK_SCALE
    begin=time.monotonic()
    base=a.base.resolve();source=a.source.resolve()
    original=json.loads((source/'manifest.json').read_text());cache=json.loads((base/'manifest.json').read_text())
    by_path={r['path']:r for r in original['records']}
    records=[];identities=[];lookup={};arrays={}
    for split in ('train','dev'):
        index=json.loads((base/(split+'-index.json')).read_text())
        for row in index['records']:
            meta=by_path[row['source']];key=(meta['object_code'],float(meta['scale']))
            if meta['split']!=split:raise ValueError('source split changed')
            if key not in lookup:lookup[key]=len(identities);identities.append(meta)
            records.append(dict(split=split,lo=row['lo'],hi=row['hi'],source=meta,geometry=lookup[key]))
        count=cache['splits'][split]['windows']
        arrays[split]=np.lib.format.open_memmap(out/(split+'-task.npy'),mode='w+',dtype=np.float32,shape=(count,10))
        for suffix in ('y.npy','tick.npy','seq.npy','index.json'):(out/(split+'-'+suffix)).symlink_to(base/(split+'-'+suffix))
    (out/'dev-panel.npy').symlink_to(base/'dev-panel.npy')
    stats=dict(np.load(base/'stats.npz'));stats['x_mean']=np.r_[stats['x_mean'],np.zeros(10,np.float32)]
    stats['x_scale']=np.r_[stats['x_scale'],TASK_SCALE];np.savez(out/'stats.npz',**stats)
    manifest=dict(schema='ref2dex.shadow-task-labels.v1',status='RUNNING',base_cache=str(base),source=str(source),
        dense=str(a.dense.resolve()),conditioning=conditioning(False),splits=cache['splits'],input_dim=310,
        git_commit=os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=sys.executable,pid=os.getpid(),workers=a.workers,records_total=len(records),records_done=0,
        inference='no neural model; causal TaskController replay only',phase_counts={s:[0,0,0] for s in arrays},
        nonzero_goal_rows={s:0 for s in arrays})
    def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    try:
        geometries=load_geometry(source,identities,dense_manifest=a.dense.resolve())
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,initargs=(str(source),records,geometries)) as pool:
            for done,(index,task) in enumerate(pool.map(read,range(len(records))),1):
                row=records[index];split=row['split']
                if task.shape!=(row['hi']-row['lo'],10):raise ValueError('row/clock alignment changed')
                arrays[split][row['lo']:row['hi']]=task
                phases=np.argmax(task[:,-3:],axis=-1)
                manifest['phase_counts'][split]=(np.asarray(manifest['phase_counts'][split])+np.bincount(phases,minlength=3)).tolist()
                manifest['nonzero_goal_rows'][split]+=int((np.linalg.norm(task[:,:6],axis=-1)>1e-7).sum())
                if done%100==0 or done==len(records):
                    for array in arrays.values():array.flush()
                    elapsed=time.monotonic()-begin
                    manifest.update(records_done=done,elapsed_s=elapsed,eta_s=elapsed/done*(len(records)-done))
                    save();print(json.dumps(dict(records_done=done,total=len(records),elapsed_s=elapsed,eta_s=manifest['eta_s'])),flush=True)
                    if (out/'STOP').exists() or elapsed>a.seconds or shutil.disk_usage(ROOT).free<20*2**30:
                        raise RuntimeError('owned stop/time/disk limit')
        manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-begin);save()
    except BaseException as e:
        manifest.update(status='FAILED',error=repr(e));save();raise


if __name__=='__main__':main()
