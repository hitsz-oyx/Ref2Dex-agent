"""TRAIN-only action-query coverage near the goal, with/without full forecast support."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np


def longest(mask):
    best=run=0
    for x in mask:
        run=run+1 if x else 0
        best=max(best,run)
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[5]
    if a.output.exists() or root/'outputs/shadow-finetune' not in a.output.resolve().parents:
        raise ValueError('fresh Task output required')
    meta=json.loads((a.cache/'manifest.json').read_text());totals=Counter();rows=[];started=time.monotonic()
    for row in meta['records']:
        if row['split']!='train':continue
        if time.monotonic()-started>600:raise TimeoutError('ten-minute coverage audit')
        with np.load(a.cache/row['path'],allow_pickle=False) as z:
            poses=z['poses'];goal=z['goal'];hand=z['hand'];actions=len(z['action'])
        distance=np.linalg.norm(poses[:,:3,3]-goal,axis=-1);near=distance<=.05
        current=np.asarray(row['ticks'],dtype=np.int64)
        # A query emits actions[t:t+8], ending in state t+8. Keep strictly before unsafe onset.
        only=np.arange(max(0,min(actions-7,row['unsafe_onset']-8)))
        local=np.einsum('tpi,tij->tpj',hand-poses[:,None,:3,3],poses[:,:3,:3])
        only=only[(local[only]>=-2).all((1,2))]
        if not set(current).issubset(only):raise ValueError('current admission is not action-only subset')
        current_near=np.zeros(len(near),bool);current_near[current]=near[current]
        only_near=np.zeros(len(near),bool);only_near[only]=near[only]
        first=np.flatnonzero(near)
        values=dict(sequences=1,current_queries=len(current),action_only_queries=len(only),
                    current_near_goal_queries=int(near[current].sum()),action_only_near_goal_queries=int(near[only].sum()),
                    sequences_with_current_near_goal=int(near[current].any()),
                    sequences_with_action_only_near_goal=int(near[only].any()),
                    sequences_with_observed_goal_hold60=int(longest(near)>=60),
                    sequences_with_current_query_hold60=int(longest(current_near)>=60),
                    sequences_with_action_only_query_hold60=int(longest(only_near)>=60))
        totals.update(values)
        rows.append(dict(raw_episode=row['raw_episode'],env=row['env'],object_code=row['object_code'],
                         first_near_goal_tick=int(first[0]) if len(first) else None,**values))
    if not rows:raise ValueError('nonempty TRAIN records required')
    firsts=[r['first_near_goal_tick'] for r in rows if r['first_near_goal_tick'] is not None]
    result=dict(status='COMPLETED',cache=str(a.cache.resolve()),split='train',totals=dict(totals),
                first_near_goal_tick_quantiles={str(q):float(np.quantile(firsts,q)) for q in (0,.25,.5,.75,1)} if firsts else {},
                elapsed_s=time.monotonic()-started,
                scope='position-only temporal coverage; alternate action-only support is hypothetical and not admitted for training')
    a.output.mkdir(parents=True)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (a.output/'sequences.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
