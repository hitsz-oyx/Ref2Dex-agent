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
