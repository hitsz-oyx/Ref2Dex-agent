#!/usr/bin/env python3
"""Read-only TRAIN census for ref3; no cleaning or training admission changes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

FIELDS = ('goal_mm', 'object_peak_mm', 'hand_peak_mm', 'hand_mean_mm',
          'target_now_mm', 'target_future_min_mm', 'target_end_mm',
          'any_future_min_mm', 'target_approach_mm')
STRATA = ('manipulation', 'contact_or_approach', 'idle_away', 'moving_far_target', 'middle_distance')


def future_min(values):
    return np.lib.stride_tricks.sliding_window_view(values[1:], 24).min(-1)


def hand_motion(hand, valid):
    n = len(hand)-24
    peak, total, count = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(1, 25):
        mask = valid[:n] & valid[k:k+n]
        error = np.linalg.norm(hand[k:k+n]-hand[:n], axis=-1).mean(-1)
        peak = np.maximum(peak, np.where(mask, error, 0).max(-1))
        total += np.where(mask, error, 0).sum(-1)
        count += mask.sum(-1)
    return peak*1000, np.divide(total*1000, count, out=np.full(n, np.nan), where=count>0)


def object_motion(points, poses):
    xyz = points[None] @ poses[:, :3, :3].transpose(0, 2, 1) + poses[:, None, :3, 3]
    n = len(poses)-24
    peak = np.zeros(n)
    for k in range(1, 25):
        error = np.linalg.norm(xyz[k:k+n]-xyz[:n], axis=-1).mean(-1)
        peak = np.maximum(peak, error)
    return error*1000, peak*1000


def surface_distance(points, poses, hand, hand_valid, pose_valid):
    # Orthogonal rigid transforms preserve Euclidean distance. Query in the
    # object's canonical frame instead of rebuilding a tree at every frame.
    local = (hand.reshape(len(hand), 22, 3)-poses[:, None, :3, 3]) @ poses[:, :3, :3]
    dist = cKDTree(points.astype(np.float64)).query(local.reshape(-1, 3), workers=1)[0].reshape(len(hand), 2, 11)
    mask = hand_valid & pose_valid[:, None]
    return np.where(mask[:, :, None], dist, np.inf).min((1, 2))*1000


def sequence_metrics(job):
    root, seq, rows = job
    path = Path(root)/'processed/sequences'/seq
    meta = json.loads((path/'meta.json').read_text())
    hand, valid, poses, pv = [np.load(path/(k+'.npy'), mmap_mode='r') for k in ('hand','hand_valid','poses','pose_valid')]
    if not np.isfinite(hand[np.repeat(valid[:, :, None], 11, 2)]).all() or not np.isfinite(poses[pv]).all():
        raise ValueError('nonfinite valid source '+seq)
    hp, hm = hand_motion(hand, valid)
    distances, motions = {}, {}
    near_differences = 0
    near = np.load(path/'near.npy', mmap_mode='r')
    anchors = set(rows[:, 1].tolist())
    for j, obj in enumerate(meta['objects']):
        with np.load(Path(root)/'processed/canonical'/(obj+'.npz')) as f:
            points = f['points']
        distances[j] = surface_distance(points, poses[:, j], hand, valid, pv[:, j])
        # Only count differences away from rounding at the historical50mm cut.
        mismatch = (distances[j]<50) != near[:, j]
        near_differences += int((mismatch & (np.abs(distances[j]-50)>.01)).sum())
        if j in anchors:
            points = points[np.linspace(0, len(points)-1, 32, dtype=np.int64)]
            motions[j] = object_motion(points, poses[:, j])
    anymin = future_min(np.min(np.stack(list(distances.values())), axis=0))
    result = np.empty((len(rows), len(FIELDS)), np.float32)
    for anchor in anchors:
        ids = np.flatnonzero(rows[:, 1]==anchor); ticks = rows[ids, 2]
        g, mo = motions[anchor]; d = distances[anchor]; fm = future_min(d)
        result[ids] = np.stack((g[ticks], mo[ticks], hp[ticks], hm[ticks],
            d[ticks], fm[ticks], d[ticks+24], anymin[ticks], d[ticks]-fm[ticks]), -1)
    if not np.isfinite(result).all():
        raise ValueError('invalid admitted-window metric '+seq)
    return result, near_differences


def masks(values, threshold=2):
    g, mo, mh, _, d0, df, _, da, approach = values.T
    near = np.minimum(d0, df)<=50
    approaching = (approach>=20)&(df<=100)
    labels = np.full(len(g), 4, np.int8)
    labels[(mh>=5)&(np.minimum(d0,df)>100)] = 3
    labels[mh<5] = 2
    labels[near|approaching] = 1
    labels[mo>=5] = 0
    return dict(goal_lt1=g<1, goal_lt2=g<2, goal_lt5=g<5,
        endpoint_static_excursion=(g<threshold)&(mo>=5),
        static_goal_near=(g<threshold)&near,
        static_goal_approach=(g<threshold)&approaching,
        static_goal_idle_away=(g<threshold)&(labels==2),
        static_goal_far_moving=(g<threshold)&(labels==3),
        static_goal_far_moving_other_near=(g<threshold)&(labels==3)&(da<=50),
        static_goal_far_from_all=(g<threshold)&(labels==3)&(da>100),
        **{'stratum/'+name:labels==i for i,name in enumerate(STRATA)})


def summarize(values, categories):
    counts = np.bincount(categories, minlength=3)
    prob = np.array([.6,.2,.2])*(counts>0); prob /= prob.sum()
    weights = prob[categories]/counts[categories]
    flags = masks(values)
    report = dict(count=len(values), category_counts=counts.tolist(), category_probabilities=prob.tolist(),
        raw={k:float(m.mean()) for k,m in flags.items()},
        sampled={k:float(weights[m].sum()) for k,m in flags.items()},
        categories={str(c):{k:float(m[categories==c].mean()) for k,m in flags.items()} for c in np.unique(categories)},
        quantiles={name:dict(zip(('p10','p50','p90','p99'),np.percentile(values[:,i],[10,50,90,99]).tolist())) for i,name in enumerate(FIELDS)})
    report['threshold_sensitivity'] = {str(t):{k:float(weights[m].sum()) for k,m in masks(values,t).items() if k.startswith('static_goal') or k.startswith('endpoint_')} for t in (1,2,5)}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    started=time.time();meta=json.loads((a.data/'processed/manifest.json').read_text());reports={}
    for descriptor in meta['sources']:
        rows=np.load(descriptor['indices']['train']['path'])
        root=Path(descriptor['root']);sm=json.loads((root/'processed/manifest.json').read_text())
        seqids=np.unique(rows[:,0]);seqids=seqids[:1] if a.smoke else seqids
        selection=np.flatnonzero(np.isin(rows[:,0],seqids));r=rows[selection]
        order=np.argsort(r[:,0],kind='stable');sortedrows=r[order]
        chunks=np.split(sortedrows,np.flatnonzero(np.diff(sortedrows[:,0]))+1)
        jobs=((str(root),sm['sequences'][int(c[0,0])],c) for c in chunks)
        metric=np.empty((len(r),len(FIELDS)),np.float32);cursor=0;near_mismatch=0
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for i,(value,mismatch) in enumerate(pool.map(sequence_metrics,jobs,chunksize=1)):
                metric[order[cursor:cursor+len(value)]]=value;cursor+=len(value);near_mismatch+=mismatch
                if i%20==0 or i==len(chunks)-1:
                    elapsed=time.time()-started
                    print(json.dumps(dict(source=descriptor['name'],sequences_done=i+1,sequences_total=len(chunks),rows_done=cursor,elapsed_s=elapsed)),flush=True)
                if time.time()-started>1200:raise TimeoutError('20min census budget')
        cats=r[:,3]-(1 if descriptor['kind']=='native' else 0)
        path=a.output/(descriptor['name']+'.npz')
        np.savez(path,indices=selection,rows=r,metrics=metric,fields=np.asarray(FIELDS),categories=cats)
        reports[descriptor['name']]=dict(summarize(metric,cats),near_mask_discrepancies=near_mismatch,source_weight=descriptor['weight'])
        (a.output/'progress.json').write_text(json.dumps(dict(status='RUNNING',sources=reports,elapsed_s=time.time()-started),indent=2))
    raw_count=sum(x['count'] for x in reports.values())
    mixed={k:sum(x['source_weight']*x['sampled'][k] for x in reports.values()) for k in next(iter(reports.values()))['sampled']}
    raw={k:sum(x['count']*x['raw'][k] for x in reports.values())/raw_count for k in mixed}
    result=dict(status='COMPLETED',engineering_only=a.smoke,split='TRAIN',rows=raw_count,fields=FIELDS,
        sources=reports,mixture_sampled=mixed,mixture_raw=raw,elapsed_s=time.time()-started,
        input_manifest=str(a.data/'processed/manifest.json'),scope='geometry audit only; no cleaning/training; surface-point proximity is not physical contact')
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status='COMPLETED',rows=raw_count,elapsed_s=result['elapsed_s'])),flush=True)


if __name__=='__main__':
    main()
