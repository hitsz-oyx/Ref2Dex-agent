"""Preregistered output swaps; R is called before these interventions."""
from dataclasses import replace
import numpy as np

COMPONENT_ARMS=('WRIST','FINGER','TIME','T0')

def swap_components(full,template,arm):
    if arm in ('A1','A2','A3','A4'):return full
    if arm not in COMPONENT_ARMS:raise ValueError('unknown intervention')
    return replace(full,
        approach_pose_object=(full if arm=='WRIST' else template).approach_pose_object.copy(),
        grasp_pose_object=(full if arm=='WRIST' else template).grasp_pose_object.copy(),
        approach_joint_angles=(full if arm=='WRIST' else template).approach_joint_angles.copy(),
        preshape_q=(full if arm=='FINGER' else template).preshape_q.copy(),
        closure_q=(full if arm=='FINGER' else template).closure_q.copy(),mode='component-'+arm)

def source_contact_frame(reference):
    force=np.linalg.norm(reference['contact_forces'][:,0],axis=-1)
    valid=np.asarray(reference['contact_force_valid']).reshape(len(force),-1)[:,0]
    contact=np.flatnonzero(valid & ((force>=.25).sum(axis=-1)>=2))
    if not len(contact):raise ValueError('reference has no valid measured two-finger contact')
    return int(contact[0])

def latch_predicted_close(consumer,plan,now):
    if not hasattr(consumer,'predicted_close_at'):raise ValueError('timing intervention not enabled')
    if not np.isfinite(consumer.predicted_close_at) and plan is not None and plan.contact_index is not None:
        consumer.predicted_close_at=float(now+(plan.contact_index+1)/30)
    return None if not np.isfinite(consumer.predicted_close_at) else consumer.predicted_close_at


def source_grasp_frame(reference,points):
    """First source acquisition with the SAME measured observer, not a tuned frame."""
    from .task_feedback import FeedbackObserver
    from .inspire_contract import NATIVE_TO_WAM
    from .pose_consumer import pose7
    observer=FeedbackObserver(points)
    for tick,obj in enumerate(reference['objects'][:,0]):
        feedback=observer.update(pose7(obj),reference['hand'][tick,0,NATIVE_TO_WAM],obj[7:13],tick/30,
            contact_forces=reference['contact_forces'][tick,0],force_valid=bool(reference['contact_force_valid'][tick,0]),workspace_ok=True)
        if feedback.force_valid and feedback.contact_observed and feedback.grasp_established:return tick
    raise ValueError('source has no force-qualified measured grasp acquisition')


def closed_loop_gt_future(local,reference,hand,object_pose,points,*,cursor=0,phase_cap=None):
    """Match native robot wrist AND fingers through acquisition; never rewind R8."""
    cap=source_grasp_frame(reference,points) if phase_cap is None else int(phase_cap)
    local=np.asarray(local);hand=np.asarray(hand);pose=np.asarray(object_pose)
    if cap+24>=len(local):raise ValueError('source acquisition needs a genuine full24 future')
    if not 0<=cursor<=cap or hand.shape!=(11,3) or pose.shape!=(4,4):raise ValueError('native11 live phase required')
    measured=(hand-pose[:3,3])@pose[:3,:3]
    errors=np.mean(np.linalg.norm(local[cursor:cap+1]-measured,axis=-1),axis=-1)
    phase=cursor+int(np.argmin(errors));indices=phase+np.arange(1,25)
    # One anchor at the matched source phase preserves the actual source future.
    # Re-anchoring every future object's pose would erase object motion (A3).
    from .pose_consumer import pose7
    source_pose=pose7(reference['objects'][phase,0])
    relative=(reference['hand'][indices,0]-source_pose[:3,3])@source_pose[:3,:3]
    return relative@pose[:3,:3].T+pose[:3,3],phase,cap
