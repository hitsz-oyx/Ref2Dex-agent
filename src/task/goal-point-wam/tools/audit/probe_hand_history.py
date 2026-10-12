#!/usr/bin/env python3
"""CPU label statistics: observed-hand forecasts on the fixed validation panel."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from goal_point_wam.data import GoalWindows
from oakink_wm.multisource import MixedWindows, MAIN_SOURCE_NAMES
from oakink_wm.data import transform_points


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    started = time.monotonic()
    parent = json.loads((args.panel.parent/'input_manifest.json').read_text())
    if digest(args.data/'processed/manifest.json') != parent['data_manifest_sha256']:
        raise ValueError('panel/data identity drift')
    base = MixedWindows(args.data, 'val')
    dataset = GoalWindows(base, goal_type='object_points')
    indices = np.load(args.panel)
    totals = {name: dict(count=0, zero=0., last_velocity=0., history_velocity=0., raw_history_max=0.)
              for name in MAIN_SOURCE_NAMES}
    windows = []
    for index in indices:
        sample = dataset[int(index)]
        sid = int(sample['source_id']); source = base.sources[sid]; name = MAIN_SOURCE_NAMES[sid]
        seq, anchor, tick, _ = map(int, source.rows[int(index)-int(base.offsets[sid])])
        raw = source.sequence(source.sequences[seq])
        observed = transform_points(raw['hand'][tick-3:tick+1].reshape(4,2,11,3), np.linalg.inv(raw['poses'][tick,anchor]))
        expected = np.concatenate(((observed[-1]-observed[-2])*30.,
                                   (observed[-1]-2*observed[-2]+observed[-3])*900., observed[-1]-observed[0]),axis=-1)
        valid_now = sample['start_hand_valid']
        error = float(np.abs(sample['start_hand_history'][valid_now]-expected[valid_now]).max())
        if error > 2e-4: raise ValueError('raw observed-history reconstruction drift')
        row = totals[name]; row['raw_history_max'] = max(row['raw_history_max'], error)
        time_steps = np.arange(1,25,dtype=np.float32)[:,None,None,None]
        history = sample['start_hand_history']
        predictions = dict(zero=np.zeros_like(sample['trajectory_target']),
                           last_velocity=time_steps/30.*history[None,...,:3],
                           history_velocity=time_steps/3.*history[None,...,6:9])
        valid = sample['trajectory_valid']; row['count'] += int(valid.sum())
        values = {}
        for key,pred in predictions.items():
            errs=np.linalg.norm(pred-sample['trajectory_target'],axis=-1)[valid]
            row[key]+=float(errs.sum());values[key]=float(errs.mean())
        windows.append(dict(source=name,mixed_index=int(index),sample_id=sample['sample_id'].tolist(),epe_m=values))
    results={}
    for name,row in totals.items():
        values={key:row[key]/row['count'] for key in ('zero','last_velocity','history_velocity')}
        values.update(raw_history_max=row['raw_history_max'],
            primary_history_improvement=1-values['history_velocity']/values['zero'],
            secondary_last_velocity_improvement=1-values['last_velocity']/values['zero'])
        values['primary_gate']=values['primary_history_improvement']>=.05
        results[name]=values
    value=dict(status='COMPLETED',schema='goal-point-wam.hand-history-screen.v1',split='val',
        computation='CPU-only observed-label statistics; no neural/model computation',
        primary_gate=sum(x['primary_gate'] for x in results.values())>=2,sources=results,windows=windows,
        elapsed_seconds=time.monotonic()-started,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        panel_sha256=digest(args.panel),data_manifest_sha256=parent['data_manifest_sha256'],
        script_sha256=digest(Path(__file__)),adapter_sha256=digest(TASK/'src/goal_point_wam/data.py'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='windows'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','panel','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
