from pathlib import Path
import numpy as np
from wam_rl_policy_v3.geometry_tau import object_local_hand
from wam_rl_policy_v3.pose_consumer import pose7
from wam_rl_policy_v3.intent_components import closed_loop_gt_future,source_grasp_frame

ROOT=Path(__file__).resolve().parents[4]

def test_preacquisition_source_state_gets_forward_closure_in_R8():
    # Actual saved robot reference: force-only contact=30, coupled grasp=50.
    path=ROOT/'src/task/ptv3-shared-wam-v6/outputs/s2b-reference-r2/reference.npz'
    with np.load(path) as z:ref={k:z[k].copy() for k in z.files}
    local=object_local_hand(ref['hand'][:,0],ref['objects'][:,0])
    # The point cloud is the same canonical geometry used by the saved source.
    import json
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records']
    source=Path(next(r['source'] for r in catalog if r['sequence']=='s1_cylindersmall_pass_1'))
    with np.load(source/'canonical.npz') as z:points=z['points'].copy()
    from wam_rl_policy_v3.task_feedback import FeedbackObserver
    from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM
    observer=FeedbackObserver(points)
    for tick in range(50):
        obj=ref['objects'][tick,0];feedback=observer.update(pose7(obj),ref['hand'][tick,0,NATIVE_TO_WAM],obj[7:13],tick/30,contact_forces=ref['contact_forces'][tick,0],force_valid=bool(ref['contact_force_valid'][tick,0]),workspace_ok=True)
    assert feedback.contact_observed and not feedback.grasp_established
    assert source_grasp_frame(ref,points)==50
    future,cursor,cap=closed_loop_gt_future(local,ref,ref['hand'][49,0],pose7(ref['objects'][49,0]),points,cursor=0)
    assert cursor+8>=50, 'R horizon rewinds before the next source acquisition while still closing'
    assert cap>=50 and cursor<=cap
    expected=local[cursor+np.arange(1,9)]@pose7(ref['objects'][49,0])[:3,:3].T+ref['objects'][49,0,:3]
    np.testing.assert_allclose(future[:8],expected,atol=2e-7)


def test_native_finger_phase_is_distinguished_at_an_identical_wrist_and_never_rewinds():
    local=np.zeros((40,11,3));tips=[2,4,6,8,10]
    local[:,tips,0]=np.arange(40)[:,None]*.01
    future,cursor,cap=closed_loop_gt_future(local,None,local[2],np.eye(4),None,cursor=0,phase_cap=3)
    assert cursor==2 and cap==3
    _,next_cursor,_=closed_loop_gt_future(local,None,local[0],np.eye(4),None,cursor=cursor,phase_cap=cap)
    assert next_cursor>=cursor
