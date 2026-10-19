"""Audit stored experts against the executor task; never changes data or tolerances."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ACTIVE=(0,1,2,3,4,5,6,8,10,12,14,15)


def audit(cache):
    meta=json.loads((cache/'manifest.json').read_text())
    fields={s:np.load(cache/f'{s}-task.npy',mmap_mode='r') for s in ('train','dev')}
    labels={s:np.load(cache/f'{s}-y.npy',mmap_mode='r') for s in fields}
    active={s:np.load(cache/f'{s}-task-active.npy',mmap_mode='r') if (cache/f'{s}-task-active.npy').exists() else None for s in fields}
    points={}
    for i,g in enumerate(meta['records']):
        path=Path(g['geometry']);path=path if path.is_absolute() else cache/path
        with np.load(path) as z:points[i]=z['query'].astype(np.float64)
    rows=[]
    for r in meta['sequences']:
        with np.load(Path(r['source'])/'trajectory.npz') as z:
            pose=z['obj'][-1].astype(np.float64);goal=z['reference_obj'][-1].astype(np.float64)
            command=z['action'][:,ACTIVE].copy()
        split=r['cache_split'];lo,hi=r['lo'],r['hi'];progress=fields[split][lo:hi,6]
        if not np.array_equal(command,labels[split][lo:hi]):raise ValueError('cached canonical labels differ: '+r['sequence'])
        p=points[r['geometry']]
        position=float(np.linalg.norm(pose[:3,3]-goal[:3,3]))
        angle=float(Rotation.from_matrix(pose[:3,:3].T@goal[:3,:3]).magnitude())
        point=float(np.linalg.norm(p@pose[:3,:3].T+pose[:3,3]-(p@goal[:3,:3].T+goal[:3,3]),axis=-1).mean())
        reached=position<=.008 and angle<=.12 and point<=.008
        mask=active[split]
        rows.append(dict(sequence=r['sequence'],split=split,commands=hi-lo,
            max_confirmed_progress=float(progress.max()),confirmed_before_last_action=bool(progress[-1]>=1-1e-6),
            final_position_m=position,final_rotation_rad=angle,final_point_m=point,final_pose_reached=reached,
            active_commands=int(mask[lo:hi].sum()) if mask is not None else None))
    summaries={}
    for split in fields:
        selected=[r for r in rows if r['split']==split]
        summaries[split]=dict(sequences=len(selected),commands=sum(r['commands'] for r in selected),
            ever_ordered_complete=sum(r['max_confirmed_progress']>=1-1e-6 for r in selected),
            final_pose_reached=sum(r['final_pose_reached'] for r in selected),
            ordered_before_last_action_and_at_final_pose=sum(r['confirmed_before_last_action'] and r['final_pose_reached'] for r in selected),
            median_final_position_m=float(np.median([r['final_position_m'] for r in selected])),
            active_commands=sum(r['active_commands'] for r in selected) if active[split] is not None else None)
    return dict(status='COMPLETED',cache=str(cache.resolve()),canonical_label_alignment='PASS',summary=summaries,rows=rows,
        interpretation='task compatibility, not BC/PPO performance; progress uses pre-action command frames; final pose uses terminal observation; final 1s hold is not inferred',
        tolerance=dict(position_m=.008,rotation_rad=.12,mean_point_m=.008),task_admission=meta.get('task_admission'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('fresh audit output required')
    report=audit(a.cache);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','task_admission')}))


if __name__=='__main__':main()
