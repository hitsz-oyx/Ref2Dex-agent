#!/usr/bin/env python3
"""Terminal label audit: retained progress versus a transient positive rise."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from analyze_targeted_contact import contrast
from run_paired_evaluator_resolution import sha


def run(args):
    import torch
    torch.set_num_threads(2)
    manifest=json.loads((args.collection/'run_manifest.json').read_text())
    if manifest['run_status']!='COMPLETED':raise ValueError('not completed')
    records=[]
    for phase in manifest['phases']:
        path=Path(phase['directory'])/'records.pt'
        if sha(path)!=phase['record_sha256']:raise ValueError('record changed')
        records.append(torch.load(path,map_location='cpu',weights_only=False))
    future=torch.cat([p['future_state'] for p in records]);state=torch.cat([p['state'] for p in records])
    contact=torch.cat([p['future_contact'] for p in records]).all(-1)
    proposal=torch.cat([p['policy_trace']['proposed_arm'] for p in records]).long()
    treated=torch.cat([p['policy_trace']['treatment'] for p in records]).bool()
    active=proposal!=4
    eligible=torch.cat([p['outcome']['drop_eligible'] for p in records])
    outcomes={k:torch.cat([p['outcome'][k] for p in records]) for k in records[0]['outcome']}
    outcomes['retained_supported_lift_mm']=(future[:,-3:,38].amin(-1)-state[:,38]).clamp_min(0)*contact[:,-3:].all(-1)*1000
    outcomes['terminal_contact']=contact[:,-3:].all(-1).float()
    episodes=[e for p in records for e in p['episode_id']]
    frames=[f'{int(m)}/{int(f)}' for p in records for m,f in zip(p['motion_id'],p['start_frame'])]
    result=dict(scope='terminal descriptive objective audit; does not refit or change the completed predeclared gate',strata={})
    for name,mask in [('all_proposals',active),('not_yet_lifted',active&~eligible),('already_lifted',active&eligible)]:
        indices=mask.nonzero().flatten().tolist()
        ep=[episodes[i] for i in indices];fr=[frames[i] for i in indices]
        result['strata'][name]={k:contrast(treated[mask],outcomes[k][mask].float(),ep,fr)
                                for k in ['supported_lift_mm','retained_supported_lift_mm','terminal_contact','drop']}
    result['positive_mean_lift_but_zero_retained']={}
    for name,mask in [('Cm',active&treated),('base',active&~treated)]:
        positive=mask&(outcomes['supported_lift_mm']>.5)
        mismatch=positive&(outcomes['retained_supported_lift_mm']<=.5)
        result['positive_mean_lift_but_zero_retained'][name]=dict(positive_windows=int(positive.sum()),transient_windows=int(mismatch.sum()),
            fraction=float(mismatch.sum()/positive.sum()) if positive.any() else None,
            dropped_with_positive_mean=int((positive&outcomes['drop'].bool()).sum()))
    if args.output.exists():raise ValueError('audit exists')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
