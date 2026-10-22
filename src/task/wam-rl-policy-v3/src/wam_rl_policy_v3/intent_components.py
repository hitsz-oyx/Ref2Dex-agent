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
