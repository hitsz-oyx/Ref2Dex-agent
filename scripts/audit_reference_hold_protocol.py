#!/usr/bin/env python3
"""Read-only reference-label audit before defining any further hold task."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.run_contact_response_probe import sha,REFERENCE


def runs(mask):
    edges=torch.diff(torch.cat((torch.zeros(1,dtype=torch.long),mask.long(),torch.zeros(1,dtype=torch.long))))
    return list(zip((edges==1).nonzero().flatten().tolist(),(edges==-1).nonzero().flatten().tolist()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() or ROOT not in output.parents:raise ValueError('unique isolated audit output required')
    original=json.loads((REFERENCE/'run_manifest.json').read_text())
    hashes=original.get('input_sha256',original.get('source_sha256',{}))
    paths=sorted(Path(path) for path in hashes if path.endswith('interaction_hand_inspire.pt'))
    if len(paths)!=3:raise ValueError('exact three reference inputs required')
    records=[]
    for path in paths:
        if sha(path)!=hashes[str(path)]:raise ValueError('reference drift')
        d=torch.load(path,map_location='cpu',weights_only=False)
        if d.ndim!=2 or d.shape[1]!=598:raise ValueError('native raw reference schema drift')
        z=d[:,200]-d[0,200];lift=z>=.03;intervals=runs(lift)
        records.append(dict(path=str(path),sha256=sha(path),frames=len(d),
            relative_object_z_column=200,contact_label_column=205,
            end_lift_mm=float(z[-1]*1000),maximum_lift_mm=float(z.max()*1000),
            lift_intervals_half_open=intervals,max_lift_run_frames=max((b-a for a,b in intervals),default=0),
            contact_lift_intervals_half_open=runs(lift&(d[:,205]>.5)),end_contact_label=float(d[-1,205])))
    episodes=[]
    for path in sorted(REFERENCE.glob('*/results.json')):
        result=json.loads(path.read_text());rows=result.get('per_episode',[])
        if rows:episodes.append(dict(path=str(path),sha256=sha(path),episodes=len(rows),stable=sum(r['stable_success'] for r in rows),
            drop_after_success=sum(r['drop_after_success'] for r in rows),retained=sum(r['stable_success'] and not r['drop_after_success'] for r in rows)))
    result=dict(status='PASS',classification='TASK_PROTOCOL_BLOCKER',reference_records=records,historical_episode_records=episodes,
        control_dt=1/30,required_hold_frames=45,planned_full_episode_retention_incompatible_with_reference_return=True,
        boundary='reference compatibility audit; no change to any already frozen experiment gate',
        reference_reader='third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py',
        source_sha256={str(path.resolve()):sha(path) for path in (Path(__file__),ROOT/'third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py',ROOT/'third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py',REFERENCE/'run_manifest.json')})
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
