"""Resumable full-pool WAM conditions aligned to the existing current BC cache."""
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .goal_scheduler import OrderedGoalScheduler
from .tau_contract import v4_trajectory_to_student


class RecordInputs(torch.utils.data.Dataset):
    def __init__(self, source, records, geometries, prepare_batch=0, task_goal=False):
        self.source, self.records, self.geometries = Path(source), records, geometries
        self.prepare_batch=prepare_batch
        self.task_goal=task_goal

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row=self.records[index];meta=row['source']
        with np.load(self.source/meta['path'],allow_pickle=False) as z:
            poses,hand,clock,obs,action,goal=(z[k] for k in ('poses','hand','time','observation','action','goal'))
        if (not np.array_equal(obs[1:,167:191],action) or
                not np.allclose(np.diff(clock),1/60,atol=1e-6) or
                not np.allclose(goal,goal[0],atol=1e-7)):
            raise ValueError('canonical labels,60Hz clock and fixed task goal required: '+meta['path'])
        ticks=np.asarray(meta['ticks'],np.int64)
        if len(ticks)!=row['hi']-row['lo'] or np.any(np.diff(ticks)<=0):
            raise ValueError('current-cache row/tick alignment changed')
        surface,_,flow,_,_=self.geometries[row['geometry']]
        target=poses[0].copy();target[:3,3]=goal[0]
        if self.task_goal:
            from .shadow_task_probe import TaskGoalScheduler
            from .shadow_task_contract import measured_velocity
            state=dict(pose=poses[:1],goal=goal[:1],velocity=measured_velocity(poses[0],obs[0])[None])
            scheduler=TaskGoalScheduler(state,0,self.geometries[row['geometry']],force_feedback=False)
        else:scheduler=OrderedGoalScheduler([poses[0],target],[0.,2.],flow,interaction_points=surface)
        goals={}
        tasks=[]
        wanted=set(ticks[ticks>=6].tolist())
        for tick in range(int(ticks[-1])+1):
            if self.task_goal:
                state['velocity']=measured_velocity(poses[tick],obs[tick])[None]
                scheduler.feedback(state,0,False)
            result=scheduler.update(poses[tick],float(clock[tick]),hand_points=hand[tick])
            if self.task_goal:tasks.append(result['task'])
            if tick in wanted:goals[tick]=result['goal_delta']
        selected=ticks[ticks>=6]
        indices=selected[:,None]+np.array([-6,-4,-2,0])[None]
        objects=np.concatenate((poses[:,:3,3],Rotation.from_matrix(poses[:,:3,:3]).as_quat()),-1)
        result=dict(index=index,ticks=ticks,selected=selected,objects=objects[indices],
                    hands=hand[indices],rotations=poses[selected,:3,:3],
                    goals=np.stack([goals[int(t)] for t in selected]) if len(selected) else np.zeros((0,32,3)))
        if self.task_goal:result['task']=np.asarray(tasks,np.float32)[ticks]
        if self.prepare_batch:
            from .wam_v4_runtime import v4_online_samples,collate_v4
            result['batches']=[]
            for lo in range(0,len(selected),self.prepare_batch):
                hi=min(lo+self.prepare_batch,len(selected))
                samples=v4_online_samples(result['objects'][lo:hi],result['hands'][lo:hi],
                    [self.geometries[row['geometry']]]*(hi-lo),goal_delta=result['goals'][lo:hi])
                result['batches'].append(collate_v4(samples))
        return result


def unchanged(value):
    return value


def prepare(args,out,write):
    from .wam_v4_runtime import V4Runtime,v4_online_samples,collate_v4
    from .bc_mlp import SCHEDULED_DIM
    from .control_contracts import scheduled_conditioning_contract
    task_goal=getattr(args,'task_goal',False)
    if task_goal:
        from .shadow_task_contract import conditioning
        from .executor_contract import TASK_SCALE
        SCHEDULED_DIM=575
    tau_start=310 if task_goal else 300
    base=args.data.resolve();source=args.sequence_cache.resolve()
    meta=json.loads((source/'manifest.json').read_text())
    cache=json.loads((base/'manifest.json').read_text())
    if meta.get('status')!='COMPLETED' or not meta.get('training_allowed') or cache['status']!='COMPLETED':
        raise ValueError('complete approved source caches required')
    by_path={r['path']:r for r in meta['records']}
    records=[]; identities=[]; geo_index={}
    for split in ('train','dev'):
        index=json.loads((base/(split+'-index.json')).read_text())
        for row in index['records']:
            src=by_path[row['source']]
            if src['split']!=split:raise ValueError('split identity changed')
            key=(src['object_code'],float(src['scale']))
            if key not in geo_index:
                geo_index[key]=len(identities);identities.append(src)
            records.append(dict(split=split,lo=row['lo'],hi=row['hi'],source=src,geometry=geo_index[key]))
    identity=dict(base=str(base),source=str(source),wam=str(args.wam_checkpoint.resolve()),
                  dense=str(args.wam_dense_manifest.resolve()),batch=args.wam_batch,
                  protocol='task-current575.v1' if task_goal else 'scheduled-current565.v1',records=len(records))
    shard_count=getattr(args,'shard_count',1);shard_index=getattr(args,'shard_index',0)
    sharded=shard_count>1
    previous=out/'cache-identity.json'
    if previous.exists() and json.loads(previous.read_text())!=identity:
        raise ValueError('cache resume input/protocol identity changed')
    if sharded and not previous.exists():raise ValueError('Coordinator must initialize cache')
    if not sharded:write('cache-identity.json',identity)
    shape=lambda split:(cache['splits'][split]['windows'],SCHEDULED_DIM)
    arrays={}
    for split in ('train','dev'):
        path=out/(split+'-x.npy')
        if path.exists():
            arrays[split]=np.load(path,mmap_mode='r+')
            if arrays[split].shape!=shape(split):raise ValueError('cache shape mismatch')
        else:
            if sharded:raise ValueError('Coordinator must initialize arrays')
            arrays[split]=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape(split))
            old=np.load(base/(split+'-x.npy'),mmap_mode='r')
            for lo in range(0,len(old),16384):arrays[split][lo:lo+16384,:300]=old[lo:lo+16384]
            arrays[split][:,300:]=0;arrays[split].flush()
        if not sharded:
            # Recopy immutable current states on resume: an interrupted initial
            # allocation may have created the file before finishing its prefix.
            old=np.load(base/(split+'-x.npy'),mmap_mode='r')
            for lo in range(0,len(old),16384):arrays[split][lo:lo+16384,:300]=old[lo:lo+16384]
            arrays[split].flush()
            for suffix in ('y.npy','tick.npy','seq.npy','index.json'):
                link=out/(split+'-'+suffix)
                if not link.exists():link.symlink_to(base/link.name)
    if not sharded:
        if not (out/'dev-panel.npy').exists():(out/'dev-panel.npy').symlink_to(base/'dev-panel.npy')
        stats=dict(np.load(base/'stats.npz'))
        stats['x_mean']=np.r_[stats['x_mean'],np.zeros(SCHEDULED_DIM-300,np.float32)]
        stats['x_scale']=np.r_[stats['x_scale'],TASK_SCALE if task_goal else np.zeros(0,np.float32),np.full(264,.05,np.float32),np.ones(1,np.float32)]
        np.savez(out/'stats.npz',**stats)
    done_path=out/('done-shard-%d.npy'%shard_index if sharded else 'done.npy')
    if sharded and not done_path.exists():raise ValueError('Coordinator must initialize shard completion map')
    if done_path.exists():done=np.load(done_path,mmap_mode='r+')
    else:
        done=np.lib.format.open_memmap(done_path,mode='w+',dtype=bool,shape=(len(records),));done[:]=False;done.flush()
    manifest=dict(schema='ref2dex.shadow-task-current-cache.v1' if task_goal else 'ref2dex.shadow-scheduled-current-cache.v1',status='RUNNING',
        source=str(source),base_cache=str(base),splits=cache['splits'],input_dim=SCHEDULED_DIM,
        conditioning=conditioning() if task_goal else scheduled_conditioning_contract(),
        condition='current300 + task10 + same-goal WAM world tau264 + history-valid1' if task_goal else 'current300 + scheduled WAM world tau264 + history-valid1',
        startup='tick<6: tau=0, valid=0; native BC still observes full current300',
        normalization='original TRAIN state/action stats; task explicit scales; tau fixed .05m; validity scale1',identity=identity,
        task_goal=task_goal)
    write('manifest.json',manifest)
    started=time.monotonic()
    runtime=V4Runtime(args.wam_checkpoint,identities,source,dense_manifest=args.wam_dense_manifest,device='cuda:0')
    assigned=[i for i in range(len(records)) if i%shard_count==shard_index]
    pending=[i for i in assigned if not done[i]]
    if args.max_records:pending=pending[:args.max_records]
    dataset=RecordInputs(source,[records[i] for i in pending],runtime.geometry,prepare_batch=args.wam_batch,task_goal=task_goal)
    loader=torch.utils.data.DataLoader(dataset,batch_size=None,num_workers=args.workers,collate_fn=unchanged,pin_memory=True,
        **(dict(multiprocessing_context='spawn',prefetch_factor=2) if args.workers else {}))
    rows_done=0;checked=False
    timings=dict(wait_s=0.,transfer_s=0.,forward_s=0.,tau_s=0.,write_s=0.);last=time.monotonic();warm_start=None
    for count,item in enumerate(loader,1):
        timings['wait_s']+=time.monotonic()-last
        if (out/'STOP').exists():raise RuntimeError('owned STOP requested; cache remains resumable')
        if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('20GiB free-space floor')
        index=pending[item['index']];row=records[index];geometry=runtime.geometry[row['geometry']]
        predicted=[]
        for lo in range(0,len(item['selected']),args.wam_batch):
            hi=min(lo+args.wam_batch,len(item['selected']))
            stage=time.monotonic()
            batch={k:v.to(runtime.device,non_blocking=True) for k,v in item['batches'][lo//args.wam_batch].items()}
            torch.cuda.synchronize();timings['transfer_s']+=time.monotonic()-stage;stage=time.monotonic()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=runtime.predict(batch)
            trajectory=prediction['trajectory'].float().cpu().numpy()
            timings['forward_s']+=time.monotonic()-stage;stage=time.monotonic()
            tau=np.stack([v4_trajectory_to_student(t,object_rotation=r) for t,r in zip(trajectory,item['rotations'][lo:hi])])
            timings['tau_s']+=time.monotonic()-stage
            if not checked:
                samples=v4_online_samples(item['objects'][lo:lo+1],item['hands'][lo:lo+1],[geometry],goal_delta=item['goals'][lo:lo+1])
                one={k:v.to(runtime.device) for k,v in collate_v4(samples[:1]).items()}
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):single=runtime.predict(one)
                reference=v4_trajectory_to_student(single['trajectory'].float().cpu().numpy(),object_rotation=item['rotations'][lo])
                error=float(np.abs(tau[:1]-reference).max())
                write('batch-check.json',dict(max_tau_difference_m=error,batch=args.wam_batch,threshold_m=.001,
                    scope='one actual batch/single check; not all-state equivalence'))
                if error>.001:raise ValueError('batch/single tau difference exceeds1mm; use --wam-batch1 in new cache')
                checked=True
            predicted.append(tau.reshape(hi-lo,264))
        stage=time.monotonic()
        block=arrays[row['split']][row['lo']:row['hi']]
        valid=item['ticks']>=6
        if task_goal:block[:,300:310]=item['task']
        if predicted:block[valid,tau_start:tau_start+264]=np.concatenate(predicted)
        block[:,-1]=valid.astype(np.float32)
        if not np.isfinite(block).all():raise ValueError('nonfinite prepared BC condition')
        arrays[row['split']].flush();done[index]=True;done.flush()
        timings['write_s']+=time.monotonic()-stage
        rows_done+=len(block)
        if warm_start is None:warm_start=time.monotonic()
        last=time.monotonic()
        if count==1 or count%10==0 or count==len(pending):
            elapsed=time.monotonic()-started;completed=int(done[assigned].sum())
            report=dict(records_done=completed,records_total=len(assigned),shard_index=shard_index,shard_count=shard_count,rows_this_run=rows_done,
                elapsed_s=elapsed,eta_s=((time.monotonic()-warm_start)/max(1,count-1)*(len(assigned)-completed)),
                warm_records_per_second=(count-1)/max(.001,time.monotonic()-warm_start),timings=timings.copy(),
                peak_mib=torch.cuda.max_memory_allocated()/2**20)
            write('progress.json',report);print(json.dumps(report),flush=True)
    manifest.update(status='COMPLETED' if done[assigned].all() else 'PARTIAL',records_done=int(done[assigned].sum()),shard_index=shard_index,shard_count=shard_count,
                    elapsed_s=time.monotonic()-started)
    write('manifest.json',manifest)
