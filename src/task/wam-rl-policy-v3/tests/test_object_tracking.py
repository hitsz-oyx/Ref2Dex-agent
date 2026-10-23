import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from wam_rl_policy_v3.object_tracking import ObjectTrajectory,make_trajectory,hand_setpoint,errors,joint_margin,SHAPES

@pytest.mark.parametrize('shape',SHAPES)
def test_reference_derivative_matches_pose_difference(shape):
    start=np.eye(4);start[:3,:3]=Rotation.from_euler('xyz',[.4,.3,-.2]).as_matrix()
    trajectory,meta=make_trajectory(shape,.02,.1,start)
    t=meta['duration_s']*.37;eps=1e-5
    pose,v,w=trajectory.evaluate(t);prev=trajectory.evaluate(t-eps)[0];after=trajectory.evaluate(t+eps)[0]
    np.testing.assert_allclose(v,(after[:3,3]-prev[:3,3])/(2*eps),atol=1e-7)
    finite=Rotation.from_matrix(after[:3,:3]@prev[:3,:3].T).as_rotvec()/(2*eps)
    np.testing.assert_allclose(w,finite,atol=1e-7)
    assert errors(trajectory.evaluate(0)[0],start)[0]<1e-12
    with pytest.raises(ValueError):trajectory.evaluate(-1)
    if meta['closed']:assert errors(trajectory.evaluate(meta['duration_s'])[0],start)[0]<1e-10


def test_rotating_offset_velocity_and_transform():
    hand_object=np.eye(4);hand_object[:3,3]=[.2,0,0]
    object_pose=np.eye(4);object_pose[:3,:3]=Rotation.from_euler('z',.4).as_matrix()
    pose,v,w=hand_setpoint(object_pose,[.1,0,0],[0,0,2],hand_object)
    np.testing.assert_allclose(pose@hand_object,object_pose,atol=1e-12)
    assert np.linalg.norm(v-[.1,0,0])==pytest.approx(.4)
    eps=1e-6;next_pose=object_pose.copy();next_pose[:3,:3]=Rotation.from_rotvec(np.asarray(w)*eps).as_matrix()@object_pose[:3,:3];next_pose[:3,3]+=[.1*eps,0,0]
    next_hand=next_pose@np.linalg.inv(hand_object)
    np.testing.assert_allclose((next_hand[:3,3]-pose[:3,3])/eps,v,atol=1e-6)


def test_continuous_wrist_not_a_limited_joint():
    mask=np.ones(18,bool);mask[3:6]=False
    limits=dict(has_limits=mask,lower=np.zeros(18),upper=np.ones(18))
    q=np.ones(18)*.5;q[3:6]=[100,-100,30];q[10]=-.12
    margins=joint_margin(q,limits)
    assert np.isinf(margins[3:6]).all();assert margins[10]==pytest.approx(-.12)
    assert np.argmin(margins)==10


def test_installed_contact_binding_without_documented_overlap():
    from wam_rl_policy_v3.object_tracking import contact_records
    contacts=np.array([(2,25,1.2)],dtype=[('body0','i4'),('body1','i4'),('lambda','f4')])
    row=contact_records(contacts)[0]
    assert row['initial_overlap'] is None
    assert row['body0']==2 and row['body1']==25 and row['force']==pytest.approx(1.2)
    richer=np.array([(2,25,1.2,.003)],dtype=contacts.dtype.descr+[('initial_overlap','f4')])
    assert contact_records(richer)[0]['initial_overlap']==pytest.approx(.003)


def test_no_feedforward_is_existing_pose_consumer_wrist_law():
    from wam_rl_policy_v3.object_tracking import ObjectPoseTracker
    from wam_rl_policy_v3.pose_consumer import PoseConsumer,WristPlan,FINGER_ACTIVE
    q=np.zeros(18);q[6:]=.3;dq=np.zeros(18);palm=np.eye(4);target=palm.copy();target[0,3]=.04
    target[:3,:3]=Rotation.from_rotvec([.02,0,0]).as_matrix()
    offset=np.zeros(18);scale=np.ones(18)
    tracker=ObjectPoseTracker(q,palm,offset,scale)
    raw,_=tracker.step(q,dq,palm,target,[0,0,0],[0,0,0],np.zeros((5,3)),False,False)
    prior=np.r_[np.zeros(6),2*q[FINGER_ACTIVE]-1]
    plan=WristPlan(np.repeat(target[None],4,axis=0),'contact_window',3,.1,np.array([1,0,0]))
    old=PoseConsumer(0,q,palm,offset,scale)
    baseline,_=old.step(q=q,dq=dq,palm=palm,object_pose=np.eye(4),forces=np.zeros((5,3)),stage={'current':{'contact':False}},plan=plan,prior_action=prior,now=0.)
    np.testing.assert_array_equal(raw,baseline)
    ff,_=tracker.step(q,dq,palm,target,[.03,0,0],[0,0,0],np.zeros((5,3)),False,True)
    assert float(ff[0]-raw[0])==pytest.approx(.03/30,abs=1e-8)


@pytest.mark.parametrize('force',[.2,5.])
def test_real_contact_finger_hold_matches_existing_consumer(force):
    from types import SimpleNamespace
    from wam_rl_policy_v3.object_tracking import ObjectPoseTracker
    from wam_rl_policy_v3.pose_consumer import MeasuredGraspConsumer,GraspIntent,FINGER_ACTIVE
    q=np.zeros(18);q[6:]=.3;dq=np.zeros(18);palm=np.eye(4);offset=np.zeros(18);scale=np.ones(18)
    forces=np.zeros((5,3));forces[:,0]=force
    tracker=ObjectPoseTracker(q,palm,offset,scale)
    actual,_=tracker.step(q,dq,palm,palm,np.zeros(3),np.zeros(3),forces,True,False)
    existing=MeasuredGraspConsumer(q,palm,offset,scale);existing.phase='hold';existing.close_pose=palm.copy();existing.fingers=q[FINGER_ACTIVE].copy()
    intent=GraspIntent(palm,palm,q[FINGER_ACTIVE],q[FINGER_ACTIVE],palm,'tracking',False)
    feedback=SimpleNamespace(force_valid=True,contact_observed=True,grasp_lost=False,grasp_established=True)
    expected,_=existing.step(q=q,dq=dq,palm=palm,object_pose=palm,object_velocity=np.zeros(6),forces=forces,feedback=feedback,
                             near_fingers=2,fingertip_gap_m=0.,intent=intent,now=1.)
    np.testing.assert_array_equal(actual[6:],expected[6:])


def test_real_runner_passes_force_valid_to_B_and_not_rigid_fixture():
    import ast
    from pathlib import Path
    runner=Path(__file__).resolve().parents[1]/'tools/run/run_object_tracking.py'
    tree=ast.parse(runner.read_text())
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='tracker' and n.func.attr=='step')
    argument=compile(ast.Expression(call.args[-2]),str(runner),'eval')
    for stage,valid,expected in [('A',True,False),('B',True,True),('B',False,False)]:
        assert bool(eval(argument,{'case':{'stage':stage},'state':{'contact_force_valid':[valid]}}))==expected


def test_handoff_keeps_actual_pd_preload_instead_of_measured_deflection():
    from wam_rl_policy_v3.object_tracking import ObjectPoseTracker
    from wam_rl_policy_v3.pose_consumer import FINGER_ACTIVE
    q=np.zeros(18);q[6:]=.3;palm=np.eye(4);held=np.array([1.36,1.05,.807,1.385,1.10,.426])
    tracker=ObjectPoseTracker(q,palm,np.zeros(18),np.ones(18),held_finger_targets=held)
    raw,_=tracker.step(q,np.zeros(18),palm,palm,np.zeros(3),np.zeros(3),np.ones((5,3)),True,False)
    reconstructed=(raw[6:]+1)/2
    np.testing.assert_allclose(reconstructed,held,atol=1e-7)
    assert abs(reconstructed[0]-q[FINGER_ACTIVE][0])>.5
