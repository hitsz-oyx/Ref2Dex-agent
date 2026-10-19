"""Full-corpus known-task fields; reuse immutable physical and WAM caches."""
import json
from pathlib import Path
import time
import numpy as np
from .inspire_shadow import CONTRACT,TASK_DIM,TASK_SLICE,INPUT_DIM,BC_ADMISSION,task_condition,NATIVE_TO_WAM


def build_task_fields(cache):
    from .goal_scheduler import OrderedGoalScheduler,update_schedulers
    from .wam_v4_runtime import load_geometry
    cache=Path(cache);meta=json.loads((cache/'manifest.json').read_text())
    geometry=load_geometry(cache,meta['records'],cache/'surface-manifest.json')
    arrays={s:np.lib.format.open_memmap(cache/f'{s}-task.npy',mode='w+',dtype=np.float32,
        shape=(meta['splits'][s]['commands'],TASK_DIM)) for s in ('train','dev')}
    active={s:np.lib.format.open_memmap(cache/f'{s}-task-active.npy',mode='w+',dtype=bool,
        shape=(meta['splits'][s]['commands'],)) for s in arrays}
    started=time.monotonic();covered=[]
    for index,row in enumerate(meta['sequences']):
        if (cache.parent/'STOP').exists():raise InterruptedError('owned task-cache stop')
        geo=geometry[row['geometry']]
        with np.load(Path(row['source'])/'trajectory.npz') as z:
            poses=z['obj'][:-1].copy();hands=z['hand'][:-1][:,NATIVE_TO_WAM].copy();reference=z['obj'].copy()
        if len(poses)!=row['hi']-row['lo']:raise ValueError('same-command task fields must align')
        scheduler=OrderedGoalScheduler(reference,np.arange(len(reference))/30,geo[2],interaction_points=geo[0],goal_type='se3_pose')
        values=[];live=[]
        for tick,pose in enumerate(poses):
            decision=update_schedulers([scheduler],pose[None],np.array([tick/30]),hands[tick:tick+1],fast_geometry=True)[0]
            values.append(task_condition(pose[None],[decision])[0])
            live.append(not decision['timed_out'])
        arrays[row['cache_split']][row['lo']:row['hi']]=values;covered.append(row['sequence'])
        active[row['cache_split']][row['lo']:row['hi']]=live
        if index%50==0:print(json.dumps(dict(stage='TASK_FIELDS',sequences=index+1,total=len(meta['sequences']),elapsed_s=time.monotonic()-started)),flush=True)
    for split,array in arrays.items():
        array.flush()
        active[split].flush()
        # Fresh full cache stores the suffix directly. Reused legacy tau arrays
        # remain read-only and are concatenated only by the trainer.
        old=np.load(cache/f'{split}-scheduled.npy',mmap_mode='r')
        if old.shape[1]==INPUT_DIM and not (cache/f'{split}-scheduled.npy').is_symlink():
            del old
            target=np.load(cache/f'{split}-scheduled.npy',mmap_mode='r+');target[:,TASK_SLICE]=array;target.flush()
    meta.update(task_status='COMPLETED',conditioning=CONTRACT,task_admission=dict(schema=BC_ADMISSION,
        semantics='exclude the timeout state and all later commands; preserve all raw source sequences',
        splits={s:dict(source_commands=len(v),active_commands=int(v.sum()),post_timeout_commands=int((~v).sum())) for s,v in active.items()}),
        current_admission='finite same-tick commands before first scheduler timeout',task_fields=dict(dim=TASK_DIM,
        semantics='measured-current-object-frame local scheduler SE3 plus confirmed progress',
        sequences=len(covered),source='known object plan + current/past measurements; no future hand',elapsed_s=time.monotonic()-started))
    (cache/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')


def reuse_full_cache(source,target):
    source=Path(source).resolve();target=Path(target).resolve()
    meta=json.loads((source/'manifest.json').read_text())
    if meta.get('conditioning',{}).get('goal_source')!=CONTRACT['goal_source']:
        raise ValueError('old reference-conditioned physical/tau cache cannot be reused for teacher-rollout goals; rebuild from audited rollout')
    if meta.get('scheduled_status')!='COMPLETED' or len(meta['sequences'])!=660 or len({r['sequence'] for r in meta['sequences']})!=660:
        raise ValueError('complete full660 physical/WAM cache required')
    if meta['splits']['train']['sequences']!=589 or meta['splits']['dev']['sequences']!=71:raise ValueError('preserve TRAIN/DEV membership')
    target.mkdir(parents=True)
    names=['stats.npz','dev-panel.npy','geometries','surface-manifest.json','wam-identity-0.json','wam-identity-1.json']
    names += [f'{s}-{k}.npy' for s in ('train','dev') for k in ('x','y','tick','seq','scheduled','include')]
    for name in names:(target/name).symlink_to(source/name)
    meta.update(source_conditioning=meta['conditioning'],conditioning=CONTRACT,task_status='PENDING',
        task_cache_source=str(source),current_cache_source=str(source),input_dim=INPUT_DIM)
    (target/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    build_task_fields(target)
