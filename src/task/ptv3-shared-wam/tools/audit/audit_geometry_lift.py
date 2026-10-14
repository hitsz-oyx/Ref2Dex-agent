"""Replay every admitted TRAIN/dev label through the branch-preserving map."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/trajectory-policy/tools/audit')]
from ptv3_shared_wam.action_targets import GEOMETRY,GEOMETRY_LIFTED,encode_targets,freeze_chunk,consume_target
from audit_competent_pd import independent_pd


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    corpus=json.loads((a.source/'manifest.json').read_text());controller=corpus['native_controller'];torch.set_num_threads(2)
    summary={};total=0
    for split in ['train','val']:
        row=dict(windows=0,old_max_PD_error=0.,new_max_PD_error=0.,old_bad_targets=0,new_bad_targets=0,new_clipped=0)
        for record in corpus['records']:
            if record['split']!=split:continue
            with np.load(a.source/record['path']/'trajectory.npz') as d:
                ticks=np.asarray(record['admitted_ticks']);query=torch.tensor(d['q'][ticks]);live=torch.tensor(d['q'][ticks[:,None]+np.arange(8)])
                pd=d['pd_targets'][ticks[:,None]+np.arange(8)]
            row['windows']+=len(ticks)
            for prefix,representation in [('old',GEOMETRY),('new',GEOMETRY_LIFTED)]:
                labels=torch.tensor(encode_targets(pd,query.numpy(),representation));frozen=freeze_chunk(labels,query,representation)
                _,applied,clipped=consume_target(frozen,live,representation,controller)
                actual=independent_pd(applied.numpy(),live.numpy(),controller)
                error=np.abs(actual-pd).max(-1)
                row[prefix+'_max_PD_error']=max(row[prefix+'_max_PD_error'],float(error.max()))
                row[prefix+'_bad_targets']+=int((error>1e-4).sum())
                if prefix=='new':row['new_clipped']+=int(clipped.sum())
        summary[split]=row;total+=row['windows']
    passed=all(v['new_max_PD_error']<5e-6 and v['new_bad_targets']==0 and v['new_clipped']==0 for v in summary.values())
    report=dict(pass_=passed,splits=summary,total_windows=total,device='cpu',CPU_reason='pure label/controller algebra audit, no neural computation',
                scope='exact labels roundtrip; predicted branch accuracy and grasp unqualified')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    if not passed:raise SystemExit('geometry lift roundtrip failed')


if __name__=='__main__':main()
