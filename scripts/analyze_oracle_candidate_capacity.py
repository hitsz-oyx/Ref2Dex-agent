"""Observed finite-bank capacity, kept separate from locked oracle utility."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);args=parser.parse_args()
    import torch
    import numpy as np
    torch.set_num_threads(2)
    out=args.source;load=lambda path:torch.load(path,map_location='cpu',weights_only=False)
    initial=load(out/'baseline/initial.pt');motion=initial['motion'].numpy()
    selector=json.loads((out/'results.json').read_text())
    state_rows=json.loads((out/'rows.json').read_text())
    attained=np.array([row['success'] for row in state_rows if row['arm']=='state'],bool)
    if len(attained)!=12 or attained.sum()!=8 or np.any(attained[motion==0]) or not np.all(attained[motion!=0]):
        raise ValueError('other eight subjects already attain capacity one')
    def label(path,target):
        trace=load(path/'trace.pt');init=load(path/'initial.pt');stop=int(init['phase_stop'][init['motion'][target]])
        mask=(np.arange(202)>=stop-74)&(np.arange(202)<=stop+30)
        if len(trace['clearance'])!=202 or mask.sum()!=105:raise ValueError('complete105')
        good=(trace['object_root'][:,target,2]-init['initial_height'][target]>=.03)&(trace['clearance'][:,target]>=.02)
        return bool(good[torch.from_numpy(mask)].all())
    table={};rows=[]
    for target in np.flatnonzero(motion==0):
        labels=[]
        for index in range(8):
            if index==0:path=out/'baseline'
            else:
                name='e%02d_c%02d'%(target,index);path=out/'deploy'/name
                if not path.exists():path=out/'capacity'/name
            success=label(path,int(target));labels.append(success)
            rows.append(dict(subject=int(target),option=index,success=success,actual_rollout=str(path)))
        table[str(target)]=labels
    available=sum(any(values) for values in table.values());ceiling=8+available
    result=dict(run_status='COMPLETED',conclusion='PROMISING' if available else 'UNPROMISING',
                observed_finite_candidate_ceiling=dict(successes=ceiling,episodes=12),
                locked_state_successes=selector['counts']['state']['successes'],
                extra_attainable_successes=available,all4_unsolved_subjects_all8_candidates=True,
                unsolved_candidate_success=table,retrospective_capacity_only=True,
                no_retrospective_policy_gain_claim=True,no_universal_Cm_upper_bound=True,
                actual_new_optimizer_steps=0,inherited_optimizer_steps=6000)
    (out/'capacity-results.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'capacity-rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
