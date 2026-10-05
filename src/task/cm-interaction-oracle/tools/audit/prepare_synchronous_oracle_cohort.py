#!/usr/bin/env python3
"""Choose two synchronous groups solely from baseline current-state support."""
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'src/task/cm-interaction-oracle/src'))
from oracle_y_utility import candidate_deltas
from intervention import all_arms_have_headroom


def main():
    root=ROOT/'outputs/cm-interaction-oracle'
    for base in ['oracle-y-utility-s263','oracle-y-utility-extra-s264']:
        folder=root/(base+'-reference')
        p=torch.load(folder/'panel.pt',weights_only=False); t=torch.load(folder/'trace.pt',weights_only=False)
        initial=torch.load(folder/'initial_state.pt',weights_only=False)
        hold=torch.zeros_like(p['triggers']); terminal=torch.zeros_like(hold,dtype=torch.bool); masks=[]
        lengths=initial['tensors']['max_episode_length'][p['motion_id']]
        rollout=initial['scalars']['rollout_length']
        for tick in range(len(t['action'])):
            phy=t['physical'][tick]; held=(phy[:,2]-p['rest_height']>=.03)&(phy[:,71]>.5)
            hold=torch.where(held,hold+1,0)
            eligible=(tick>=9)&(hold>=6)&(hold<45)&(phy[:,66:71].amin(-1)<.06)&~terminal
            eligible &= (lengths-p['start_frame']-tick>91)&(rollout-tick>91)
            eligible &= all_arms_have_headroom(t['action'][tick],candidate_deltas())
            masks.append(eligible);terminal|=t['done'][tick]
        masks=torch.stack(masks);non=(p['motion_id']!=0);best=(-1,None)
        for first in range(len(masks)):
            one=masks[first]; remaining=masks&~one
            support=remaining.sum(1);support[~(remaining&non).any(1)]=-1
            second=int(support.argmax());count=int(one.sum()+support[second])
            if count>best[0]:best=count,(first,second,one,remaining[second])
        count,(a,b,one,two)=best
        trigger=torch.full_like(p['triggers'],-1);groups=trigger.clone()
        trigger[one]=a;trigger[two]=b;groups[one]=0;groups[two]=1
        out=root/(base+'-sync-schedule.json')
        value=dict(schema='ref2dex.synchronous_candidate_schedule.v1',triggers=trigger.tolist(),groups=groups.tolist(),
            clocks=[a,b],sizes=[int(one.sum()),int(two.sum())],anchors=count,
            motion_counts=[int(((one|two)&(p['motion_id']==m)).sum()) for m in range(3)],
            selection='Maximize two disjoint current-eligible groups; group2 requires non-s3; lexicographic earliest clock ties.',
            eligibility='current6<=hold<45,history10,forcepair,proximity<6cm,all headroom,horizon>91,no previous native terminal',
            no_future_Y_or_Z_or_alternative_outcomes_used=True,
            input_sha256={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in (folder/'panel.pt',folder/'trace.pt',folder/'initial_state.pt')})
        with out.open('x') as f:json.dump(value,f,indent=2)
        print(json.dumps(dict(schedule=str(out),clocks=value['clocks'],sizes=value['sizes'],anchors=count,motion_counts=value['motion_counts'])))
if __name__=='__main__':main()
