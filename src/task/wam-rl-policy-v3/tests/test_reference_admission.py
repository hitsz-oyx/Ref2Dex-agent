import numpy as np
import pytest
from wam_rl_policy_v3.reference_admission import JointEnvelope, ProgressWatchdog, UnreachableReference, limit_categories
from wam_rl_policy_v3.tau_replay_contract import pd_targets


def fixture():
    mask=np.ones(18,bool);mask[3:6]=False
    lo=np.zeros(18);lo[:2]=-5
    hi=np.ones(18)*1.6;hi[:3]=[5,5,3];hi[14]=1.15;hi[15]=.55
    bounds=dict(has_limits=mask.tolist(),lower=lo.tolist(),upper=hi.tolist())
    scale=hi.copy();scale[:3]=1;scale[3:6]=np.pi
    q=np.zeros(18);q[2]=1;q[6:]=.3;q[16]=.18;q[17]=.24
    palm=np.eye(4);palm[2,3]=1
    return JointEnvelope(bounds,np.zeros(18),scale),q,palm,bounds


def test_unreachable_floating_position_is_refused_before_action():
    env,q,palm,_=fixture();target=palm.copy();target[0,3]=6
    with pytest.raises(UnreachableReference):env.pose_configuration(target,q,palm,q[env.fingers])


def test_passive_thumb_margin_limits_driver_before_reference_selection():
    env,q,palm,bounds=fixture();env.lower[16]=.18
    fingers=q[env.fingers].copy();fingers[-1]=.25
    with pytest.raises(UnreachableReference):env.pose_configuration(palm,q,palm,fingers)


def test_continuous_wrist_can_rotate_across_two_pi():
    env,q,palm,_=fixture();q[3]=8*np.pi
    target=env.pose_configuration(palm,q,palm,q[env.fingers])
    assert abs(target[3]-q[3])<1e-8


def test_command_projection_preserves_legal_finger_preload_and_all_mimics():
    env,q,palm,_=fixture();a=np.zeros(12);a[6:]=-.2
    old=pd_targets(a,q,env.offset,env.scale)
    command,info=env.protect(a,q,np.zeros(18))
    np.testing.assert_allclose(pd_targets(command,q,env.offset,env.scale),old,atol=1e-7)
    assert not info['braking']


def test_outward_actual_momentum_stops_root_even_when_target_is_legal():
    env,q,palm,_=fixture();q[16]=.005;dq=np.zeros(18);dq[16]=-1
    a=np.zeros(12);a[0]=.02;a[6:]=-.2
    command,info=env.protect(a,q,dq)
    assert info['braking'] and 16 in info['risk_joints']
    assert command[0]==pytest.approx(0,abs=1e-8)
    assert env.check(pd_targets(command,q,env.offset,env.scale))==[]


def test_three_limit_categories_are_not_conflated():
    env,q,palm,bounds=fixture();q[6]=-.01;q[8]=.02
    target=q.copy();target[10]=1.7
    x=limit_categories(q,target,bounds)
    assert x['true_outside']==[6]
    assert 8 in x['near_inside'] and 6 not in x['near_inside']
    assert x['command_outside_state_inside']==[10]
    assert not any(j in x['true_outside'] for j in (3,4,5))


def test_stationary_pose_error_eventually_stops_but_moving_state_does_not():
    target=np.eye(4);target[0,3]=.1;actual=np.eye(4)
    w=ProgressWatchdog();status=None
    for t in np.arange(0,1.1,1/30):status=w.update(t,actual,target)
    assert status=='UNREACHABLE_STALL'
    w=ProgressWatchdog()
    for t in np.arange(0,1.1,1/30):
        actual[0,3]=.03*t
        assert w.update(t,actual,target) is None


def test_intermediate_reference_excursion_is_rejected_even_if_endpoints_are_legal():
    from wam_rl_policy_v3.object_tracking import ObjectTrajectory
    env,q,palm,_=fixture();poses=np.repeat(palm[None],3,axis=0);poses[1,0,3]=6
    curve=ObjectTrajectory([0,.5,1],poses)
    decision=env.admit_trajectory(curve,np.eye(4),q,palm,q[env.fingers])
    assert not decision['admitted'] and decision['time_s']<1


def test_a_different_palm_kinematic_chain_is_not_silently_certified():
    env,q,palm,_=fixture();palm[0,3]=.05
    with pytest.raises(UnreachableReference,match='calibration'):env.pose_configuration(palm,q,palm,q[env.fingers])


def test_actual_track_path_checks_reference_and_records_final_safe_pd():
    from wam_rl_policy_v3.object_tracking import ObjectPoseTracker
    env,q,palm,bounds=fixture();c=ObjectPoseTracker(q,palm,env.offset,env.scale,held_finger_targets=q[env.fingers],joint_limits=bounds)
    target=palm.copy();target[0,3]=6
    with pytest.raises(UnreachableReference):c.step(q,np.zeros(18),palm,target,np.zeros(3),np.zeros(3),np.zeros((5,3)),False,True)
    a,info=c.step(q,np.zeros(18),palm,palm,np.zeros(3),np.zeros(3),np.zeros((5,3)),False,True)
    assert info['admission'] is not None
    np.testing.assert_allclose(info['admission']['executed_pd'],pd_targets(a,q,env.offset,env.scale))


def test_bandwidth_fit_recovers_known_delay_gain_and_discards_static_offset():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'tools/audit/audit_limit_tracking.py'
    spec=importlib.util.spec_from_file_location('limit_bandwidth_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    t=np.arange(151)/30
    def reference(x):return np.stack([np.sin(x*2),np.cos(x*3),np.sin(x*1.3)],axis=-1)
    fit=module.lag_gain(t,.7*reference(t-.2)+[1,2,3],reference)
    assert fit['delay_s']==pytest.approx(.2)
    assert fit['gain']==pytest.approx(.7)


def test_disabling_experimental_braking_keeps_mandatory_native_admission():
    env,q,palm,_=fixture();q[16]=.005;dq=np.zeros(18);dq[16]=-1
    action=np.zeros(12,np.float32);action[0]=.02;action[6:]=-.2
    unchanged,info=env.protect(action,q,dq,predictive_braking=False)
    np.testing.assert_array_equal(unchanged,action)
    assert not info['braking']
    action[-1]=1. # thumb pitch at upper nominal boundary has no inner margin
    protected,info=env.protect(action,q,dq,predictive_braking=False)
    assert info['target_projected']
    assert env.check(pd_targets(protected,q,env.offset,env.scale))==[]
