import numpy as np
from scipy.spatial.transform import Rotation
from wam_rl_policy_v3.pose_consumer import (pose7,object_relative_reference,live_reference_future,
    reduce_to_wrist_plan,PoseConsumer,closest_wrist_angles)


def hand():
    rng=np.random.RandomState(1)
    return rng.normal(size=(11,3))*.025


def test_gt_target_moves_with_live_object_and_phase_uses_measurement_not_clock():
    h=np.repeat(hand()[None],40,axis=0);h[:,:,0]+=np.linspace(0,.3,40)[:,None]
    objects=np.zeros((40,7));objects[:,6]=1
    local=object_relative_reference(h,objects)
    pose=np.eye(4);pose[:3,3]=[.3,.2,.1]
    live=local[12]+pose[:3,3]
    future,phase=live_reference_future(local,live,pose,contact_frame=30)
    assert phase==12
    np.testing.assert_allclose(future[0],h[13]+pose[:3,3])
    shifted=pose.copy();shifted[:3,3]+=.4
    other,phase2=live_reference_future(local,live+.4,shifted,contact_frame=30)
    assert phase2==12
    np.testing.assert_allclose(other,future+.4)


def test_rigid_wrist_extraction_and_no_contact_approach_preserve_real_future():
    h=hand()+[1,0,0];palm=np.eye(4);palm[:3,3]=h[0]
    transform=np.eye(4);transform[:3,:3]=Rotation.from_rotvec([.1,.02,-.03]).as_matrix();transform[:3,3]=[.01,0,0]
    future=np.repeat((h@transform[:3,:3].T+transform[:3,3])[None],24,axis=0)
    plan=reduce_to_wrist_plan(h,palm,future,np.eye(4),np.zeros((4,3)))
    assert plan.mode=='approach' and plan.contact_index is None and plan.eta_s>=.8
    np.testing.assert_allclose(plan.local_poses[-1],transform@palm,atol=1e-7)
    assert np.linalg.norm(plan.approach_direction_object)>0


def test_measured_contact_controls_switch_and_postcontact_plan_is_ignored():
    q=np.zeros(18);palm=np.eye(4);q[6:]=.2
    c=PoseConsumer(0,q,palm,np.zeros(18),np.ones(18))
    stage=dict(current=dict(contact=False))
    args=dict(q=q,dq=np.zeros(18),palm=palm,object_pose=np.eye(4),forces=np.ones((5,3)),prior_action=np.zeros(12))
    _,info=c.step(**args,stage=stage,plan=None,now=0)
    assert info['phase']=='approach'
    stage['current']['contact']=True
    _,info=c.step(**args,stage=stage,plan=None,now=1)
    assert info['phase']=='close'
    _,info=c.step(**args,stage=stage,plan=None,now=2)
    assert info['phase']=='lift_hold' and info['wrist_target'][2][3]>.05


def test_wrist_inverse_preserves_measured_mount_and_nearest_continuous_angles():
    q=np.array([-1.7,-3.4,2.2]);mount=Rotation.from_euler('xyz',[3.14,0,1.57]).as_matrix()
    target=Rotation.from_euler('XYZ',q).as_matrix()@mount
    result=closest_wrist_angles(target,q,mount)
    np.testing.assert_allclose(result,q,atol=1e-10)


def test_approach_mode_consumes_eta_direction_and_does_not_invent_contact():
    from wam_rl_policy_v3.pose_consumer import WristPlan
    palm=np.eye(4);path=np.repeat(palm[None],24,axis=0);path[:,0,3]=.1
    def command(eta):
        c=PoseConsumer(0,np.zeros(18),palm,np.zeros(18),np.ones(18))
        return c.step(q=np.zeros(18),dq=np.zeros(18),palm=palm,object_pose=palm,
            forces=np.zeros((5,3)),stage=dict(current=dict(contact=False)),
            plan=WristPlan(path,'approach',None,eta,np.array([1.,0,0])),
            prior_action=np.zeros(12),now=0)
    fast,info=command(1.);slow,_=command(2.)
    assert fast[0]>slow[0]>0 and info['phase']=='approach'
    assert info['predicted_contact_index'] is None
