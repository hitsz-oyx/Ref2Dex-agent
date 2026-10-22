"""Physical contract regressions: measured grasp-relative SE3 and native limits."""
import numpy as np
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from wam_rl_policy_v3.grasp_repair import RepairedGraspConsumer
from wam_rl_policy_v3.pose_consumer import GraspIntent


def held_step(consumer_class=RepairedGraspConsumer,extra=None):
    q=np.zeros(18);q[6:]=.2;palm=np.eye(4);palm[2,3]=1.2
    obj=np.eye(4);obj[2,3]=1.1;goal=obj.copy();goal[:3,:3]=Rotation.from_euler('z',.5).as_matrix()
    points=np.array([[x,y,z] for x in (-.02,.02) for y in (-.02,.02) for z in (-.02,.02)])
    c=consumer_class(q,palm,np.zeros(18),np.ones(18),plane=(np.array([0,0,1.]),.9),palm_vertices=np.zeros((1,3)),object_points=points,**(extra or {}))
    c.phase='lift';c.close_pose=palm.copy();c.fingers=np.ones(6)*.2
    intent=GraspIntent(np.linalg.inv(obj)@palm,np.linalg.inv(obj)@palm,np.ones(6)*.2,np.ones(6)*.2,goal,'held',True,False)
    feedback=SimpleNamespace(force_valid=True,contact_observed=True,grasp_lost=False,grasp_established=True)
    command,info=c.step(hand=np.tile(palm[:3,3],(11,1)),q=q,dq=np.zeros(18),palm=palm,object_pose=obj,object_velocity=np.zeros(6),forces=np.ones((5,3)),feedback=feedback,near_fingers=2,fingertip_gap_m=0.,intent=intent,now=0.)
    return command,info,c


def test_postgrasp_pose_command_contains_object_orientation_error():
    from wam_rl_policy_v3.grasp_geometry import GeometryGraspConsumer
    limits=dict(has_limits=[True]*3+[False]*3+[True]*12,lower=[-5,-5,0]+[0]*15,upper=[5,5,3]+[3.2]*15)
    command,info,_=held_step(GeometryGraspConsumer,dict(joint_limits=limits))
    assert command[5]>.01, 'Object yaw error must enter the grasp-relative wrist servo'
    assert info['phase']=='lift', 'Position alone cannot certify a pose task'


def limits():
    return dict(has_limits=[True]*3+[False]*3+[True]*12,lower=[-5,-5,0]+[0]*15,upper=[5,5,3]+[3.2]*15)


def test_geometry_thresholds_scale_and_are_rigid_transform_invariant():
    from wam_rl_policy_v3.grasp_geometry import GraspGeometry
    points=np.array([[x,y,z] for x in (-.02,.02) for y in (-.03,.03) for z in (-.04,.04)])
    a=GraspGeometry.from_points(points);b=GraspGeometry.from_points(points*2)
    c=GraspGeometry.from_points(points@Rotation.from_rotvec([.2,.3,.7]).as_matrix().T+[1,2,3])
    assert abs(a.position_tolerance_m-.004)<1e-10
    assert abs(b.position_tolerance_m-2*a.position_tolerance_m)<1e-10
    assert abs(c.position_tolerance_m-a.position_tolerance_m)<1e-10
    assert a.arrived(.003) and not a.arrived(.005)
    assert not a.arrived(.003,.5,False)


def test_passive_joint_limit_rejects_candidate_before_ranking():
    from wam_rl_policy_v3.grasp_geometry import JointAdmission,grasp_candidates
    bounds=limits();bounds['upper'][7]=.1
    admission=JointAdmission(bounds,np.zeros(18),np.ones(18));q=np.zeros(18)
    raw=np.r_[np.zeros(6),np.ones(6)*-.6] # target finger .2 -> passive .21 > .1
    okay,bad=admission.check(raw,q)
    assert not okay and 7 in bad
    p=np.eye(4);p[2,3]=1.;obj=p.copy();hand=np.random.RandomState(2).normal(size=(11,3))*.02+p[:3,3]
    points=np.array([[x,y,z] for x in (-.02,.02) for y in (-.02,.02) for z in (-.02,.02)])
    candidate,info=grasp_candidates(p,hand,obj,points,np.array([0,0,1.]),.9,p,np.ones(6)*.2,q,admission)
    assert candidate is None and info['admitted']==0 and info['rejected_limits']==info['count']
    bounded=admission.bound_action(raw,q)
    assert admission.check(bounded,q)[0]


def test_relative_transform_includes_rotational_lever_arm_and_norm_rate_limits():
    from wam_rl_policy_v3.grasp_geometry import RelativePoseServo
    s=RelativePoseServo();obj=np.eye(4);palm=np.eye(4);palm[0,3]=.1
    goal=np.eye(4);goal[:3,:3]=Rotation.from_euler('z',np.pi/2).as_matrix();goal[2,3]=.08
    target,angles,info=s.step(goal,obj,palm,np.zeros(18),np.zeros(18),np.eye(3),0.,False)
    desired=np.array(info['desired_wrist'])
    np.testing.assert_allclose(desired[:3,3],[0,.1,.08],atol=1e-10)
    assert info['linear_step_m']<=.75/30+1e-10
    assert info['angular_step_rad']<=2.4/30+1e-10
    assert np.linalg.norm(target[:3,3]-palm[:3,3])<=.025+1e-10
    s.reset();assert s.grasp_relative is None


def test_two_finger_measurement_ignores_three_distant_nonparticipating_fingers():
    from wam_rl_policy_v3.grasp_geometry import GraspGeometry
    points=np.array([[x,y,z] for x in (-.02,.02) for y in (-.02,.02) for z in (-.02,.02)])
    g=GraspGeometry.from_points(points);hand=np.ones((11,3));hand[6]=points[0];hand[7]=points[-1]
    assert g.paired_tip_gap(hand,np.eye(4),points)==0
    hand[7]=[1,1,1]
    assert g.paired_tip_gap(hand,np.eye(4),points)>g.tip_gap_m


def test_residual_is_bounded_in_intent_and_does_not_rotate_translation_about_object():
    from wam_rl_policy_v3.intent_residual import apply_intent_residual,RESIDUAL_SCALE
    p=np.eye(4);p[:3,3]=[.1,.2,.3]
    intent=GraspIntent(p.copy(),p.copy(),np.ones(6)*.2,np.ones(6)*.3,np.eye(4),'x',True)
    changed=apply_intent_residual(intent,np.ones(12)*100)
    np.testing.assert_allclose(changed.approach_pose_object[:3,3]-p[:3,3],RESIDUAL_SCALE[:3],atol=1e-8)
    np.testing.assert_allclose(changed.preshape_q-intent.preshape_q,RESIDUAL_SCALE[6:],atol=1e-8)
    assert np.linalg.norm(Rotation.from_matrix(changed.approach_pose_object[:3,:3]).as_rotvec())<=np.linalg.norm(RESIDUAL_SCALE[3:6])+1e-8
    np.testing.assert_array_equal(intent.approach_pose_object,p)
    assert changed.approach_joint_angles is None


def test_actual_residual_ppo_uses_time_limit_bootstrap_and_has_bounded_epochs():
    import torch
    from wam_rl_policy_v3.intent_residual import IntentResidualPolicy,ppo_update
    torch.set_num_threads(2);torch.manual_seed(293)
    policy=IntentResidualPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
    obs=np.zeros((2,698),np.float32)
    with torch.no_grad():
        dist,_=policy.distribution(torch.tensor(obs));latent=dist.sample();old=dist.log_prob(latent).sum(-1)
    result=ppo_update(policy,optimizer,obs,latent.numpy(),old.numpy(),[0.,1.],[0.,0.],bootstrap=2.)
    # Last return=1+.96*2; previous GAE=.96*.95*last.
    expected=(1+.96*2)*(1+.96*.95)/2
    assert abs(result['target_return_mean']-expected)<1e-5
    assert 1<=result['optimizer_updates']<=4
    assert all(np.isfinite(e['loss']) for e in result['epochs'])


def test_relative_servo_still_corrects_measured_object_slip_after_wrist_arrived():
    from wam_rl_policy_v3.grasp_geometry import RelativePoseServo
    servo=RelativePoseServo();goal=np.eye(4);obj=np.eye(4);palm=np.eye(4);palm[0,3]=.1
    servo.step(goal,obj,palm,np.zeros(18),np.zeros(18),np.eye(3),0.,True)
    obj[0,3]=.02
    target,_,_=servo.step(goal,obj,palm,np.zeros(18),np.zeros(18),np.eye(3),1/30,True)
    assert target[0,3]<palm[0,3]-.001, 'Measured object drift must remain in feedback after the captured wrist goal is reached'
