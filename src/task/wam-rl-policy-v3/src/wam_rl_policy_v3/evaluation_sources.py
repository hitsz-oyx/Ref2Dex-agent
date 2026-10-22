"""Frozen v8 controls: absent future, one geometric template, and S2d GT targets."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from .pose_consumer import GraspIntent,pose7,closest_wrist_angles,FINGER_ACTIVE,FINGER_UPPER


def continuous_hold(times, held):
    """Independent elapsed-time scorer; disjoint bouts never accumulate."""
    times=np.asarray(times,float);held=np.asarray(held,bool)
    if times.ndim!=1 or held.shape!=times.shape or not np.isfinite(times).all() or (np.diff(times)<=0).any():
        raise ValueError('aligned strictly increasing physical sample times required')
    start=None;longest=last=0.
    for i,(now,valid) in enumerate(zip(times,held)):
        if i and now-times[i-1]>1/30+1e-8:start=None
        if valid:
            if start is None:start=now
            last=float(now-start);longest=max(longest,last)
        else:start=None;last=0.
    return dict(longest_hold_s=longest,ever_success=bool(longest>=1-1e-10),final_success=bool(len(held) and held[-1] and last>=1-1e-10))


def absent_tau(kind,available):
    if kind not in ('zero-valid','no-tau'):raise ValueError('explicit absence arm required')
    return np.zeros((8,11,3),np.float32),bool(available and kind=='zero-valid')


@dataclass
class FixedGraspTemplate:
    pregrasp_pose_object:np.ndarray
    grasp_pose_object:np.ndarray
    preshape_q:np.ndarray
    closure_q:np.ndarray


def freeze_template(reference,contact_frame,held_frame):
    """One declared robot demonstration, no WAM trajectory or learned alignment."""
    grasp=np.linalg.inv(pose7(reference['objects'][held_frame,0]))@pose7(reference['palm_pose'][held_frame,0])
    pre=grasp.copy();direction=pre[:3,3]/max(np.linalg.norm(pre[:3,3]),1e-12);pre[:3,3]+=.04*direction
    before=max(int(contact_frame)-3,0)
    preshape=np.clip(reference['q'][before,0,FINGER_ACTIVE],.06,FINGER_UPPER)
    closure=np.clip(reference['q'][held_frame,0,FINGER_ACTIVE]+.015,.06,FINGER_UPPER)
    return FixedGraspTemplate(pre,grasp,preshape,closure)


def template_intent(template,q,palm,object_pose,object_goal,phase):
    """Current object + fixed keys -> the same measured consumer, without R/WAM."""
    pre=object_pose@template.pregrasp_pose_object;grasp=object_pose@template.grasp_pose_object
    desired=pre if phase=='approach' and np.linalg.norm(pre[:3,3]-palm[:3,3])>.025 else grasp
    mount=Rotation.from_euler('XYZ',q[3:6]).as_matrix().T@palm[:3,:3]
    angles=closest_wrist_angles(desired[:3,:3],q[3:6],mount)
    angles=q[3:6]+np.clip(angles-q[3:6],-.12,.12)
    target=np.eye(4);target[:3,:3]=Rotation.from_euler('XYZ',angles).as_matrix()@mount
    target[:3,3]=palm[:3,3]+np.clip(desired[:3,3]-palm[:3,3],-.025,.025)
    return GraspIntent(np.linalg.inv(object_pose)@target,template.grasp_pose_object.copy(),
        template.preshape_q.copy() if phase=='approach' else template.closure_q.copy(),
        template.closure_q.copy(),np.asarray(object_goal).copy(),'fixed-template',False,
        approach_joint_angles=angles)


def gt_future(s2d_targets,reference,frame,live_object,available):
    """Reuse S2d's whole-future object anchor; future[0] is t+1/30s."""
    if not available or frame<3 or frame+24>=len(reference['hand']):return None,False
    target=np.asarray(s2d_targets(reference['hand'][:,0],reference['objects'][:,0],frame,live_object,horizon=24),np.float32)
    return target,True
