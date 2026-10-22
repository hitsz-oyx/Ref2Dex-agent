from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from wam_rl_policy_v3.intent_components import swap_components,latch_predicted_close,source_contact_frame
from wam_rl_policy_v3.pose_consumer import live_reference_future
from test_grasp_intent import setup

@pytest.mark.parametrize('arm',['WRIST','FINGER','TIME','T0'])
def test_output_swaps_remove_both_keyposes_and_euler_branch_without_goal_leak(arm):
    _,args,_,_=setup();full=args['intent'];template=replace(full,approach_pose_object=np.eye(4),grasp_pose_object=np.eye(4)*2,approach_joint_angles=np.ones(3)*3,preshape_q=np.ones(6)*.4,closure_q=np.ones(6)*.8)
    got=swap_components(full,template,arm)
    for name in ('approach_pose_object','grasp_pose_object','approach_joint_angles'):
        np.testing.assert_array_equal(getattr(got,name),getattr(full if arm=='WRIST' else template,name))
    for name in ('preshape_q','closure_q'):
        np.testing.assert_array_equal(getattr(got,name),getattr(full if arm=='FINGER' else template,name))
    assert got.tau_valid==full.tau_valid
    np.testing.assert_array_equal(got.postgrasp_object_pose,full.postgrasp_object_pose)
    got.preshape_q[0]=99
    assert full.preshape_q[0]!=99 and template.preshape_q[0]!=99

def test_joint_full_arm_is_identity_and_zero_tau_control_is_not_fixed_template():
    _,args,_,_=setup();full=args['intent']
    for arm in ('A1','A2','A3','A4'):assert swap_components(full,None,arm) is full

def test_timing_only_suppresses_measured_ready_until_deadline_but_cannot_invent_lift():
    c,args,_,_=setup();args['near_fingers']=5;c.predicted_close_at=float('inf')
    assert c.step(**args,now=0)[1]['phase']=='approach'
    deadline=latch_predicted_close(c,SimpleNamespace(contact_index=1),1/30)
    assert deadline==pytest.approx(.1)
    latch_predicted_close(c,SimpleNamespace(contact_index=0),2/30)
    assert c.predicted_close_at==deadline
    assert c.step(**args,now=1/30)[1]['phase']=='approach'
    assert c.step(**args,now=2/30)[1]['phase']=='approach'
    assert c.step(**args,now=3/30)[1]['phase']=='close'
    for tick in range(4,20):assert c.step(**args,now=tick/30)[1]['phase']=='close'

def test_source_contact_uses_valid_measured_two_finger_force_and_rejects_absence():
    r=dict(contact_forces=np.zeros((5,1,5,3)),contact_force_valid=np.zeros((5,1),bool));r['contact_forces'][:,:,:,0]=1;r['contact_force_valid'][3]=True
    assert source_contact_frame(r)==3
    r['contact_force_valid'][:]=False
    with pytest.raises(ValueError):source_contact_frame(r)

def test_closed_loop_gt_cursor_does_not_expire_with_wall_clock_or_run_backwards():
    local=np.ones((60,11,3));local[:,:,0]=np.arange(60)[:,None]*.01
    future,cursor=live_reference_future(local,local[8],np.eye(4),contact_frame=10,cursor=0)
    assert cursor==8 and future.shape==(24,11,3)
    future,cursor=live_reference_future(local,local[0],np.eye(4),contact_frame=10,cursor=cursor)
    assert cursor==8
