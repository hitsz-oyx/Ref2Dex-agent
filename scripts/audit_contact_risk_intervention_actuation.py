#!/usr/bin/env python3
"""Independent scalar PD comparison on observed trajectories, no counterfactual Y."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def run(args):
    import hashlib
    import torch
    torch.set_num_threads(2)
    begin=time.monotonic()
    if args.output.exists():raise ValueError('unique actuation audit output')
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=json.loads((args.source/'run_manifest.json').read_text())
    result=json.loads(args.results.read_text())
    if manifest['run_status']!='COMPLETED' or result['run_status']!='COMPLETED':raise ValueError('terminal source/analysis')
    count=steps=0;hashes={str(args.results.resolve()):sha(args.results)}
    for phase in manifest['phases']:
        path=Path(phase['directory'])/'records.pt'
        if sha(path)!=phase['result']['record_sha256']:raise ValueError('actual record hash')
        hashes[str(path.resolve())]=sha(path)
        record=torch.load(path,map_location='cpu',weights_only=False)
        for row in (record['assignment']==2).nonzero().flatten().tolist():
            weights=record['candidate_weights'][row,3].double().numpy()
            changed=[]
            for tick in range(10):
                bank=record['expert_bank'][row,tick].double().numpy()
                action=bank[1].copy()
                for node,(start,stop) in enumerate(((0,3),(6,8),(8,10),(10,12),(12,14),(14,18))):
                    for channel in range(start,stop):action[channel]=sum(float(weights[node,e])*float(bank[e,channel]) for e in range(6))
                q=record['state'][row,:18].double().numpy() if tick==0 else record['future_state'][row,tick-1,:18].double().numpy()
                offset,scale=record['pd_offset'].double().numpy(),record['pd_scale'].double().numpy()
                for channel in (3,4,5):action[channel]=np.clip((float(record['hold_target'][row,channel])-q[channel]-offset[channel])/scale[channel],-1,1)
                target=np.zeros(18)
                for channel in range(18):
                    target[channel]=offset[channel]+scale[channel]*(action[channel] if channel<6 else (action[channel]+1)/2)
                    if channel<6:target[channel]+=q[channel]
                for dst,src,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
                    target[dst]=target[src]*ratio
                changed.append(float(np.max(np.abs(target-record['actual_pd_targets'][row,tick].double().numpy())))>1e-5)
            count+=int(any(changed));steps+=sum(changed)
    wanted=result['coverage']
    if count!=wanted['actual_cm_different_direct_windows'] or steps!=wanted['actual_cm_different_direct_steps']:
        raise ValueError('independent actual PD change counts')
    audit=dict(run_status='COMPLETED',audit_passed=True,actual_cm_changed_windows=count,actual_cm_changed_steps=steps,
        inputs_sha256=hashes,auditor_sha256=sha(Path(__file__)),elapsed_seconds=time.monotonic()-begin,
        scope='scalar alternative PD on each same actual Cm observation; no unexecuted physics or regret claim')
    args.output.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k!='inputs_sha256'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
