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
    def __init__(self, source, records, geometries):
        self.source, self.records, self.geometries = Path(source), records, geometries

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
        scheduler=OrderedGoalScheduler([poses[0],target],[0.,2.],flow,interaction_points=surface)
        goals={}
        wanted=set(ticks[ticks>=6].tolist())
        for tick in range(int(ticks[-1])+1):
            result=scheduler.update(poses[tick],float(clock[tick]),hand_points=hand[tick])
            if tick in wanted:goals[tick]=result['goal_delta']
        selected=ticks[ticks>=6]
        indices=selected[:,None]+np.array([-6,-4,-2,0])[None]
        objects=np.concatenate((poses[:,:3,3],Rotation.from_matrix(poses[:,:3,:3]).as_quat()),-1)
        return dict(index=index,ticks=ticks,selected=selected,objects=objects[indices],
                    hands=hand[indices],rotations=poses[selected,:3,:3],
                    goals=np.stack([goals[int(t)] for t in selected]) if len(selected) else np.zeros((0,32,3)))


def unchanged(value):
    return value


def prepare(args,out,write):
    from .wam_v4_runtime import V4Runtime,v4_online_samples,collate_v4
    from .bc_mlp import SCHEDULED_DIM
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
                  protocol='scheduled-current565.v1',records=len(records))
    previous=out/'cache-identity.json'
    if previous.exists() and json.loads(previous.read_text())!=identity:
        raise ValueError('cache resume input/protocol identity changed')
    write('cache-identity.json',identity)
    shape=lambda split:(cache['splits'][split]['windows'],SCHEDULED_DIM)
    arrays={}
    for split in ('train','dev'):
        path=out/(split+'-x.npy')
        if path.exists():
            arrays[split]=np.load(path,mmap_mode='r+')
            if arrays[split].shape!=shape(split):raise ValueError('cache shape mismatch')
        else:
            arrays[split]=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape(split))
            old=np.load(base/(split+'-x.npy'),mmap_mode='r')
            for lo in range(0,len(old),16384):arrays[split][lo:lo+16384,:300]=old[lo:lo+16384]
            arrays[split][:,300:]=0;arrays[split].flush()
        # Recopy immutable current states on resume: an interrupted initial
        # allocation may have created the file before finishing its prefix.
        old=np.load(base/(split+'-x.npy'),mmap_mode='r')
        for lo in range(0,len(old),16384):arrays[split][lo:lo+16384,:300]=old[lo:lo+16384]
        arrays[split].flush()
        for suffix in ('y.npy','tick.npy','seq.npy','index.json'):
            link=out/(split+'-'+suffix)
            if not link.exists():link.symlink_to(base/link.name)
    if not (out/'dev-panel.npy').exists():(out/'dev-panel.npy').symlink_to(base/'dev-panel.npy')
    stats=dict(np.load(base/'stats.npz'))
    stats['x_mean']=np.r_[stats['x_mean'],np.zeros(265,np.float32)]
    stats['x_scale']=np.r_[stats['x_scale'],np.full(264,.05,np.float32),np.ones(1,np.float32)]
    np.savez(out/'stats.npz',**stats)
    done_path=out/'done.npy'
    if done_path.exists():done=np.load(done_path,mmap_mode='r+')
    else:
        done=np.lib.format.open_memmap(done_path,mode='w+',dtype=bool,shape=(len(records),));done[:]=False;done.flush()
    manifest=dict(schema='ref2dex.shadow-scheduled-current-cache.v1',status='RUNNING',
        source=str(source),base_cache=str(base),splits=cache['splits'],input_dim=SCHEDULED_DIM,
        condition='current300 + scheduled WAM world tau264 + history-valid1',
        startup='tick<6: tau=0, valid=0; native BC still observes full current300',
        normalization='original TRAIN state/action stats; tau fixed .05m; validity scale1',identity=identity)
    write('manifest.json',manifest)
    started=time.monotonic()
    runtime=V4Runtime(args.wam_checkpoint,identities,source,dense_manifest=args.wam_dense_manifest,device='cuda:0')
    pending=[i for i in range(len(records)) if not done[i]]
    if args.max_records:pending=pending[:args.max_records]
    dataset=RecordInputs(source,[records[i] for i in pending],runtime.geometry)
    loader=torch.utils.data.DataLoader(dataset,batch_size=None,num_workers=args.workers,collate_fn=unchanged,
        **(dict(multiprocessing_context='spawn',prefetch_factor=2) if args.workers else {}))
    rows_done=0;checked=False
    for count,item in enumerate(loader,1):
        if (out/'STOP').exists():raise RuntimeError('owned STOP requested; cache remains resumable')
        if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('20GiB free-space floor')
        index=pending[item['index']];row=records[index];geometry=runtime.geometry[row['geometry']]
        predicted=[]
        for lo in range(0,len(item['selected']),args.wam_batch):
            hi=min(lo+args.wam_batch,len(item['selected']))
            samples=v4_online_samples(item['objects'][lo:hi],item['hands'][lo:hi],[geometry]*(hi-lo),
                                      goal_delta=item['goals'][lo:hi])
            batch={k:v.to(runtime.device) for k,v in collate_v4(samples).items()}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=runtime.predict(batch)
            trajectory=prediction['trajectory'].float().cpu().numpy()
            tau=np.stack([v4_trajectory_to_student(t,object_rotation=r) for t,r in zip(trajectory,item['rotations'][lo:hi])])
            if not checked:
                one={k:v.to(runtime.device) for k,v in collate_v4(samples[:1]).items()}
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):single=runtime.predict(one)
                reference=v4_trajectory_to_student(single['trajectory'].float().cpu().numpy(),object_rotation=item['rotations'][lo])
                error=float(np.abs(tau[:1]-reference).max())
                write('batch-check.json',dict(max_tau_difference_m=error,batch=args.wam_batch,threshold_m=.001,
                    scope='one actual batch/single check; not all-state equivalence'))
                if error>.001:raise ValueError('batch/single tau difference exceeds1mm; use --wam-batch1 in new cache')
                checked=True
            predicted.append(tau.reshape(hi-lo,264))
        block=arrays[row['split']][row['lo']:row['hi']]
        valid=item['ticks']>=6
        if predicted:block[valid,300:564]=np.concatenate(predicted)
        block[:,564]=valid.astype(np.float32)
        if not np.isfinite(block).all():raise ValueError('nonfinite prepared BC condition')
        arrays[row['split']].flush();done[index]=True;done.flush()
        rows_done+=len(block)
        if count==1 or count%10==0 or count==len(pending):
            elapsed=time.monotonic()-started;completed=int(done.sum())
            report=dict(records_done=completed,records_total=len(records),rows_this_run=rows_done,
                elapsed_s=elapsed,eta_s=elapsed/count*(len(records)-completed),
                peak_mib=torch.cuda.max_memory_allocated()/2**20)
            write('progress.json',report);print(json.dumps(report),flush=True)
    manifest.update(status='COMPLETED' if done.all() else 'PARTIAL',records_done=int(done.sum()),
                    elapsed_s=time.monotonic()-started)
    write('manifest.json',manifest)
