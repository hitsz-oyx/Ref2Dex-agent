import numpy as np
from shadow_finetune.metrics import native_report


def test_peak_lift_outside_hold_cannot_pass_simultaneous_hold():
    obj=np.zeros((100,1,7));obj[:20,0,0]=1;obj[10,0,2]=.2;obj[20:,0,2]=.05
    d=dict(object=obj,goal=np.broadcast_to([0,0,.05],(100,1,3)),success=np.ones((100,1)))
    r=native_report(d,[dict(split='dev',object_code='object')])
    assert r['official_success']==1 and r['final_within5cm']==1
    assert r['longest_goal_hold_frames']==[80] and r['max_lift_m']==[.2]
    assert r['held60_with_lift10cm']==0 and r['by_split']['dev']['episodes']==1
