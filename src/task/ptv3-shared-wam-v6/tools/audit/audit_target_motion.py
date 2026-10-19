"""Read-only motion scan of actually admitted windows; never edits frozen data.

Center motion is independent of CAD scale and rigid world-frame choice. This
is a diagnostic, not an automatic admission policy or contact-quality label.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import os
import sys
import time
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v5.storage import read_json, load_array
from ptv3_shared_wam_v6.recipe import validate_recipe


def window_motion(poses, valid, rows, observed):
    """Return center displacement and observed-transition speeds for H4/K24."""
    rows = np.asarray(rows, np.int64)
    slots, ticks = rows[:, 1], rows[:, 2]
    translation = np.asarray(poses[..., :3, 3], np.float64)
    future = ticks[:, None] + np.arange(1, 25)
    center = translation[future, slots[:, None]] - translation[ticks, slots][:, None]
    maximum = np.linalg.norm(center, axis=-1).max(1)
    # Derive time from explicit observed flags at the preserved30Hz grid.
    ends = np.flatnonzero(observed)
    starts, ends = ends[:-1], ends[1:]
    speed = np.linalg.norm(translation[ends]-translation[starts], axis=-1) * (30./(ends-starts))[:, None]
    known = valid[starts] & valid[ends]
    speed = np.where(known, speed, 0.)
    peak = np.zeros(len(rows)); tail5 = np.zeros(len(rows), bool)
    for slot in np.unique(slots):
        select = np.flatnonzero(slots == slot)
        for first in range(0, len(select), 512):
            ids = select[first:first+512]
            overlap = (ends[None] > ticks[ids, None]-3) & (starts[None] < ticks[ids, None]+24)
            peak[ids] = np.where(overlap, speed[None, :, slot], 0.).max(1) if len(ends) else 0.
        flags = ends[speed[:, slot] > 5.]
        if len(flags): tail5[select] = ticks[select]+24 >= int(flags[0])
    return maximum, peak, tail5


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seconds',type=int,default=900)
    p.add_argument('--sources',default='0,1,2,4,5,6,8,9,10')
    p.add_argument('--git-commit',required=True)
    a=p.parse_args()
    if not 1<=a.workers<=8 or not 1<=a.seconds<=1800:raise ValueError('bounded workers/runtime required')
    recipe=validate_recipe(a.manifest);meta=read_json(recipe['parent_manifest'])
    selected=set(map(int,a.sources.split(',')));a.output.mkdir(exist_ok=False)
    started=time.monotonic();indices={};positions={};strata={};totals={};top=[];done=0
    for split in ('train','dev'):
        index=load_array(meta['indices'][split],mmap_mode='r');indices[split]=index
        order=np.argsort(index[:,0],kind='stable');cut=np.r_[0,np.cumsum(np.bincount(index[:,0],minlength=len(meta['sequences'])))]
        positions[split]=[order[cut[i]:cut[i+1]] for i in range(len(meta['sequences']))]
        strata[split]=load_array(meta['strata'][split],mmap_mode='r')
    jobs=[(i,r) for i,r in enumerate(meta['sequences']) if r['source_id'] in selected]
    def one(job):
        seqid,row=job;folder=Path(row['path'])
        pose=load_array(folder/'object_poses.npy',mmap_mode='r')
        valid=load_array(folder/'object_pose_valid.npy',mmap_mode='r')
        observed=load_array(folder/'observed_frame.npy',mmap_mode='r')
        records=[]
        for split in ('train','dev'):
            ids=positions[split][seqid]
            if not len(ids):continue
            windows=indices[split][ids];displacement,speed,tail=window_motion(pose,valid,windows,observed)
            masks={'center_gt_'+str(cap)+'m':displacement>cap for cap in (.5,1.,2.,5.)}
            masks.update(center_speed_gt5mps=speed>5.,center_speed_gt10mps=speed>10.,shadow_tail_after5mps=tail if row['source_id']==4 else np.zeros(len(ids),bool))
            groups={}
            for cat in (-1,0,1,2) if row['source_id']<3 else (-1,):
                keep=np.ones(len(ids),bool) if cat==-1 else strata[split][ids]==cat
                groups[str(cat)]=dict(windows=int(keep.sum()),**{k:int((v&keep).sum()) for k,v in masks.items()})
            worst=int(displacement.argmax())
            records.append(dict(split=split,groups=groups,maximum_center_displacement_m=float(displacement[worst]),maximum_center_speed_mps=float(speed.max()),worst_sample=windows[worst].tolist(),worst_index=int(ids[worst])))
        return dict(key=row['key'],source_id=row['source_id'],seq_id=seqid,records=records)
    status='COMPLETED';failures=[]
    with (a.output/'sequences.jsonl').open('x') as stream,ThreadPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(one,jobs):
            stream.write(json.dumps(result)+'\n');done+=1
            for rec in result['records']:
                key=str(result['source_id'])+':'+rec['split'];total=totals.setdefault(key,{})
                for group,counts in rec['groups'].items():
                    bucket=total.setdefault(group,{k:0 for k in counts})
                    for name,count in counts.items():bucket[name]+=count
                top.append(dict(key=result['key'],source_id=result['source_id'],**{k:v for k,v in rec.items() if k!='groups'}))
            top=sorted(top,key=lambda r:r['maximum_center_displacement_m'],reverse=True)[:40]
            if done%500==0:
                progress=dict(status='RUNNING',done=done,total=len(jobs),elapsed_s=time.monotonic()-started)
                (a.output/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
            if time.monotonic()-started>a.seconds or (a.output/'STOP').exists():status='PARTIAL';break
    exposure={}
    for split in ('train','dev'):
        if split=='dev':continue
        for sid in selected:
            groups=totals.get(str(sid)+':'+split)
            if not groups:continue
            allcounts=groups['-1'];prob=meta['source_probabilities'][str(sid)]
            if sid<3:
                active=[k for k in range(3) if groups[str(k)]['windows']];weight=np.array([.6,.2,.2]);denom=sum(weight[k] for k in active)
                ratio={k:sum(weight[g]/denom*groups[str(g)][k]/groups[str(g)]['windows'] for g in active) for k in allcounts if k!='windows'}
            else:ratio={k:v/allcounts['windows'] for k,v in allcounts.items() if k!='windows'}
            exposure[str(sid)]=dict(source_probability=prob,within_source_bad_fraction=ratio,global_draw_contribution={k:prob*v for k,v in ratio.items()})
    report=dict(status=status,git_commit=a.git_commit,manifest=str(a.manifest),done=done,total=len(jobs),elapsed_s=time.monotonic()-started,totals=totals,train_exposure=exposure,top=top,
        limitations=['Center translations only; rotation/surface-point speed not scanned.','Threshold exceedance is diagnostic, not automatically an erroneous human label.','Full TRAIN exposure exact only if status COMPLETED; main3 source-stratum weights included.','No future targets or recipes modified.'])
    for name in ('summary.json','progress.json'):
        path=a.output/name;path.write_text(json.dumps(report,indent=2)+'\n');path.chmod(0o664)
    print(json.dumps(dict(status=status,done=done,elapsed_s=report['elapsed_s'])),flush=True)


if __name__=='__main__':main()
