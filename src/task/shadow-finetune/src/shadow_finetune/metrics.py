"""Descriptive native success and position-hold proxies, split by object identity."""
import numpy as np


def longest_true(mask):
    best=run=0
    for x in mask:
        run=run+1 if x else 0;best=max(best,run)
    return best


def native_report(data,identities):
    distance=np.linalg.norm(data['object'][...,:3]-data['goal'],axis=-1)
    lift_now=data['object'][...,2]-data['object'][0,:,2]
    held=np.asarray([longest_true(distance[:,i]<=.05) for i in range(len(identities))])
    held_lift=np.asarray([longest_true((distance[:,i]<=.05)&(lift_now[:,i]>=.1)) for i in range(len(identities))])
    flags=dict(official_success=data['success'][-1]>0,final_within5cm=distance[-1]<=.05,
               held60_with_lift10cm=held_lift>=60)
    def aggregate(ids):return dict(episodes=len(ids),**{k:int(v[ids].sum()) for k,v in flags.items()})
    report=aggregate(list(range(len(identities))))
    report.update(longest_goal_hold_frames=held.tolist(),longest_goal_hold_with_lift_frames=held_lift.tolist(),
                  max_lift_m=lift_now.max(0).tolist(),final_goal_distance_m=distance[-1].tolist())
    for kind in ('split','object_code'):
        report['by_'+kind]={value:aggregate([i for i,r in enumerate(identities) if r[kind]==value])
                           for value in sorted({r[kind] for r in identities})}
    return report
