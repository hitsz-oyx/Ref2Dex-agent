from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from wam_rl_policy_v3.pose_consumer import (MeasuredGraspConsumer,grasp_intent_from_retargeter,GraspIntent)
from wam_rl_policy_v3.tau_replay_contract import pd_targets


def setup():
    q=np.zeros(18);q[3:6]=[-1.7,-.9,2.]
    q[6:]=.2;palm=np.eye(4);palm[:3,:3]=Rotation.from_euler('XYZ',q[3:6]).as_matrix()
    scale=np.ones(18);scale[3:6]=np.pi;offset=np.zeros(18)
    goal=np.eye(4);goal[2,3]=.08
    intent=grasp_intent_from_retargeter(q,palm,np.eye(4),goal,np.linspace(-.03,.03,12),offset,scale,tau_valid=True)
    c=MeasuredGraspConsumer(q,palm,offset,scale)
    args=dict(q=q,dq=np.zeros(18),palm=palm,object_pose=np.eye(4),object_velocity=np.zeros(6),forces=np.zeros((5,3)),
        feedback=SimpleNamespace(force_valid=True,contact_observed=False,grasp_lost=False,grasp_established=False),near_fingers=0,fingertip_gap_m=1.,intent=intent)
    return c,args,scale,offset


def test_three_part_interface_preserves_wrist_and_finger_action_geometry_and_task_goal():
    c,args,scale,offset=setup();raw=np.linspace(-.03,.03,12)
    command,info=c.step(**args,now=0)
    np.testing.assert_allclose(pd_targets(command,args['q'],offset,scale),pd_targets(raw,args['q'],offset,scale),atol=1e-6)
    np.testing.assert_allclose(args['intent'].postgrasp_object_pose[:3,3],[0,0,.08])
    assert info['phase']=='approach' and len(info['preshape_q'])==6 and len(info['closure_q'])==6


def test_elapsed_time_or_predicted_mode_cannot_switch_phase_without_measured_position():
    c,args,_,_=setup();args['intent'].mode='contact_window'
    for t in (0,1,100):assert c.step(**args,now=t)[1]['phase']=='approach'


def test_measured_position_closes_but_invalid_force_never_starts_lift():
    c,args,_,_=setup();args['near_fingers']=2
    assert c.step(**args,now=0)[1]['phase']=='close'
    args['feedback'].contact_observed=True;args['feedback'].force_valid=False
    for t in np.arange(1,10)/30:assert c.step(**args,now=t)[1]['phase']=='close'


def test_measured_contact_dwell_starts_object_goal_servo_and_object_error_changes_sign():
    c,args,_,_=setup();args['near_fingers']=2;args['feedback'].contact_observed=True;args['feedback'].grasp_established=True
    args['forces'][:2,0]=1
    for t in np.arange(5)/30:command,info=c.step(**args,now=t)
    assert info['phase']=='lift' and command[2]>0
    args['object_pose'][2,3]=.15
    command,_=c.step(**args,now=5/30)
    assert command[2]<0
    args['feedback'].contact_observed=False
    assert c.step(**args,now=6/30)[1]['phase']=='close'


def test_hand_preshape_change_is_preserved_and_no_contact_time_can_override_invalid_position():
    c,args,scale,offset=setup();args['intent'].preshape_q[0]+=.2
    command,_=c.step(**args,now=0)
    assert abs(pd_targets(command,args['q'],offset,scale)[6]-args['intent'].preshape_q[0])<1e-6
    with pytest.raises(ValueError,match='strictly increasing'):c.step(**args,now=0)


def test_wrist_ik_branch_cannot_disagree_with_pose_and_wide_action_keeps_original_solution():
    c,args,scale,offset=setup();raw=np.linspace(-.8,.8,12)
    args['intent']=grasp_intent_from_retargeter(args['q'],args['palm'],args['object_pose'],np.eye(4),raw,offset,scale)
    command,_=c.step(**args,now=0)
    np.testing.assert_allclose(command,raw,atol=1e-6)
    args['intent'].approach_joint_angles[0]+=.2
    with pytest.raises(ValueError,match='IK branch'):c.step(**args,now=1/30)


def test_live_same_tick_wrist_mount_is_used_instead_of_reset_mount():
    c,args,scale,offset=setup()
    args['palm'][:3,:3]=Rotation.from_rotvec([2e-4,0,0]).as_matrix()@args['palm'][:3,:3]
    raw=np.linspace(-.03,.03,12)
    args['intent']=grasp_intent_from_retargeter(args['q'],args['palm'],args['object_pose'],np.eye(4),raw,offset,scale)
    command,_=c.step(**args,now=0)
    np.testing.assert_allclose(command,raw,atol=1e-6)


def test_closing_keeps_reaching_until_measured_force_bearing_grasp_then_servos_goal():
    c,args,_,_=setup();args['near_fingers']=2
    args['intent'].approach_pose_object[0,3]=.02
    command,info=c.step(**args,now=0)
    assert info['phase']=='close' and command[0]>.01
    args['feedback'].contact_observed=True
    command,info=c.step(**args,now=1/30)
    assert info['phase']=='close' and command[0]>.01
    args['feedback'].grasp_established=True
    for t in np.arange(2,6)/30:command,info=c.step(**args,now=t)
    assert info['phase']=='lift' and abs(command[0])<1e-8


def test_close_preserves_live_finger_preshape_until_measured_grasp_is_established():
    c,args,scale,offset=setup();args['near_fingers']=2
    args['feedback'].grasp_established=False
    args['intent'].preshape_q[0]=.9
    command,info=c.step(**args,now=0)
    assert info['phase']=='close'
    assert abs(pd_targets(command,args['q'],offset,scale)[6]-.9)<1e-6
    args['intent'].preshape_q[0]=.7
    command,_=c.step(**args,now=1/30)
    assert abs(pd_targets(command,args['q'],offset,scale)[6]-.7)<1e-6
