"""Report the canonical native660 position prebuffer subset without changing raw data."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v2.paths import DEFAULT_ROBOT_SOURCE
from ptv3_shared_wam_v2.rollout_admission import DEFAULT_POLICY
from ptv3_shared_wam_v2.robot_data import V2RLRobotWindows
from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=DEFAULT_ROBOT_SOURCE)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('fresh report output required')
    workspace=dict(schema='observed-fixed-grid.v1',origin_m=[-2.,-2.,-2.],voxel_m=.01)
    reports={}
    for split in ('train','val'):
        ds=V2RLRobotWindows(a.source,split,32,GEOMETRY_LIFTED,workspace,DEFAULT_POLICY)
        rows=[]
        for seq,ticks in enumerate(ds.ticks_by_sequence):
            data,_=ds.sequence(seq)
            err=np.linalg.norm(data['obj'][:,:3,3]-data['reference_obj'][:,:3,3],axis=-1)
            bad=np.flatnonzero(~np.isfinite(err)|(err>.1))
            # Independent interval search, not the implementation's difference array.
            for tick in ticks:
                candidates=bad[(bad>=max(int(tick)-3,0))&(bad<=int(tick)+24+50)]
                if len(candidates):raise AssertionError((ds.records[seq]['sequence'],int(tick)))
            rows.append(dict(sequence=ds.records[seq]['sequence'],windows=len(ticks),
                             startup=int((ticks<16).sum()),admitted_ticks=ticks.tolist()))
        reports[split]=dict(sequences=len(ds.records),nonempty_sequences=sum(r['windows']>0 for r in rows),
            windows=len(ds),startup=sum(r['startup'] for r in rows),
            rollout_admission=ds.rollout_admission,workspace_admission=ds.workspace_admission,records=rows)
        print(json.dumps(dict(split=split,windows=len(ds),startup=reports[split]['startup'],
                             nonempty_sequences=reports[split]['nonempty_sequences'])),flush=True)
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(dict(source=str(a.source.resolve()),policy=DEFAULT_POLICY,
         independent_check='PASS',splits=reports),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
