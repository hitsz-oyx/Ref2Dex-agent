"""Object-relative precontact wrist plans and a measured-state grasp controller.

No predicted fingertip position is tracked by the controller. Future hand points
are reduced to rigid wrist poses, a contact time, and an explicit approach mode.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from .inspire_contract import ACTIVE

POSE_CONSUMER = 'object-wrist.precontact.measured-close-lift.v1'
RIGID_BASES = np.array([0, 1, 3, 5, 7, 9])
TIPS = np.array([2, 4, 6, 8, 10])


def pose7(value):
    value = np.asarray(value)
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(value[3:7]).as_matrix()
    result[:3, 3] = value[:3]
    return result


def object_relative_reference(hand, objects):
    """Each recorded hand is attached to its OWN recorded object frame."""
    hand, objects = np.asarray(hand), np.asarray(objects)
    rotations = Rotation.from_quat(objects[:, 3:7]).as_matrix()
    return np.einsum('tki,tij->tkj', hand-objects[:, None, :3], rotations)


def live_reference_future(local_reference, live_hand, live_object, *, contact_frame, cursor=0):
    """Measured geometry selects phase; map targets through the live object SE3."""
    local_hand = (np.asarray(live_hand)-live_object[:3, 3]) @ live_object[:3, :3]
    end = min(int(contact_frame), len(local_reference)-1)
    start = min(int(cursor), end)
    errors = np.mean(np.linalg.norm(local_reference[start:end+1, RIGID_BASES]
                                      - local_hand[RIGID_BASES], axis=-1), axis=-1)
    phase = start + int(np.argmin(errors))
    indices = np.minimum(phase+np.arange(1, 25), len(local_reference)-1)
    # This is an absolute pose target, not fabricated displacement supervision.
    future = local_reference[indices] @ live_object[:3, :3].T + live_object[:3, 3]
    return future, phase


@dataclass
class WristPlan:
    local_poses: np.ndarray
    mode: str
    contact_index: object
    eta_s: float
    approach_direction_object: np.ndarray


def reduce_to_wrist_plan(live_hand, palm, future_hand, object_pose, object_points):
    live_hand, future_hand = np.asarray(live_hand), np.asarray(future_hand)
    if future_hand.shape != (24, 11, 3) or live_hand.shape != (11, 3):
        raise ValueError('full24 future and measured native11 required')
    if not np.isfinite(np.r_[live_hand.ravel(), future_hand.ravel()]).all():
        raise ValueError('finite measured and forecast hands required')
    before = live_hand[RIGID_BASES]; center = before.mean(axis=0)
    candidates = []
    for future in future_hand:
        after = future[RIGID_BASES]; target_center = after.mean(axis=0)
        u, _, vt = np.linalg.svd((before-center).T @ (after-target_center))
        rotation = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
        delta = np.eye(4); delta[:3, :3] = rotation
        delta[:3, 3] = target_center-rotation@center
        candidates.append(np.linalg.inv(object_pose) @ delta @ palm)
    candidates = np.asarray(candidates)
    future_local = (future_hand-object_pose[:3, 3]) @ object_pose[:3, :3]
    gaps = np.linalg.norm(future_local[:, TIPS, None]-object_points[None, None], axis=-1).min(axis=-1)
    contact = np.flatnonzero((gaps <= .025).sum(axis=1) >= 2)
    index = int(contact[0]) if len(contact) else None
    # Stop at the predicted contact pose; no post-contact hand trajectory enters.
    path = candidates[:index+1] if index is not None else candidates
    displacement = path[-1, :3, 3]-(np.linalg.inv(object_pose)@palm)[:3, 3]
    length = np.linalg.norm(displacement)
    direction = displacement/max(length, 1e-12)
    speed = max(length/(len(path)/30), .02)
    eta = (index+1)/30 if index is not None else .8 + max(float(np.partition(gaps[-1], 1)[1])-.025, 0.)/speed
    return WristPlan(path, 'contact_window' if index is not None else 'approach', index, float(eta), direction)


def closest_wrist_angles(target_rotation, measured_q, base_rotation):
    angles = Rotation.from_matrix(target_rotation @ base_rotation.T).as_euler('XYZ')
    alternatives = np.array([angles, [angles[0]+np.pi, np.pi-angles[1], angles[2]+np.pi]])
    alternatives += 2*np.pi*np.round((measured_q-alternatives)/(2*np.pi))
    return alternatives[np.argmin(np.linalg.norm(alternatives-measured_q, axis=-1))]


class PoseConsumer:
    """Approach by pose; measured contact switches to closure and vertical lift."""
    def __init__(self, initial_z, q, palm, offset, scale):
        self.initial_z = float(initial_z)
        self.offset, self.scale = np.asarray(offset), np.asarray(scale)
        self.base_rotation = Rotation.from_euler('XYZ', q[3:6]).as_matrix().T @ palm[:3, :3]
        self.phase = 'approach'; self.contact_time = None
        self.hold_pose = None; self.fingers = None
        self.last_target = palm.copy()

    def step(self, *, q, dq, palm, object_pose, forces, stage, plan, prior_action, now):
        q, dq = np.asarray(q), np.asarray(dq)
        # The stage signal is supplied by S2c, never recreated here.
        measured_contact = bool(stage['current']['contact'])
        if measured_contact and self.contact_time is None:
            self.contact_time = float(now); self.hold_pose = palm.copy()
            self.fingers = q[np.asarray(ACTIVE)[6:]].copy()
            self.phase = 'close'
        contact_force = np.linalg.norm(forces, axis=-1)
        if self.contact_time is not None:
            self.phase = 'close' if now-self.contact_time < .20 else 'lift_hold'
            target = self.hold_pose.copy()
            if self.phase == 'lift_hold':
                target[2, 3] += min(.10, .075*max(now-self.contact_time-.20, 0.))
            # Hold force-bearing fingers; add bounded compression on weak ones.
            finger_force = np.r_[contact_force[:4],contact_force[4],contact_force[4]]
            increments = np.where(finger_force < .65, .008, np.where(finger_force > 3., -.002, 0.))
            self.fingers += increments
            upper = np.array([1.55,1.55,1.55,1.55,1.10,.50])
            self.fingers = np.clip(self.fingers, .06, upper)
            finger_action = 2*(self.fingers-self.offset[np.asarray(ACTIVE)[6:]])/self.scale[np.asarray(ACTIVE)[6:]]-1
        else:
            target = self.last_target.copy()
            if plan is not None:
                # Receding wrist pose, never a tracked fingertip trajectory.
                if plan.mode == 'approach':
                    target = object_pose @ plan.local_poses[-1]
                    distance = np.linalg.norm(target[:3,3]-palm[:3,3])
                    travel = min(distance, distance/max(plan.eta_s, 1/30)*4/30)
                    target[:3,3] = palm[:3,3] + object_pose[:3,:3]@plan.approach_direction_object*travel
                else:
                    target = object_pose @ plan.local_poses[min(3, len(plan.local_poses)-1)]
            finger_action = np.asarray(prior_action)[6:].copy()
            # At the wrist goal, allow closure to create measured contact. An
            # estimated contact event alone never advances measured phase.
            if plan is not None and np.linalg.norm(target[:3,3]-palm[:3,3]) < .015 and plan.mode=='contact_window':
                desired = q[np.asarray(ACTIVE)[6:]] + .012
                desired = np.clip(desired,.06,np.array([1.55,1.55,1.55,1.55,1.10,.50]))
                finger_action = 2*(desired-self.offset[np.asarray(ACTIVE)[6:]])/self.scale[np.asarray(ACTIVE)[6:]]-1
        self.last_target = target.copy()
        translation = np.clip(.45*(target[:3,3]-palm[:3,3])-.012*dq[:3], -.025, .025)
        angles = closest_wrist_angles(target[:3,:3], q[3:6], self.base_rotation)
        angular = np.clip(.40*(angles-q[3:6])-.008*dq[3:6], -.12, .12)
        raw = np.r_[(translation-self.offset[:3])/self.scale[:3],
                    (angular-self.offset[3:6])/self.scale[3:6],finger_action].astype(np.float32)
        return raw, dict(phase=self.phase, plan_mode=None if plan is None else plan.mode,
                         contact_eta_s=None if plan is None else plan.eta_s,
                         predicted_contact_index=None if plan is None else plan.contact_index,
                         direction_object=None if plan is None else plan.approach_direction_object.tolist(),
                         wrist_target=target.tolist(), measured_contact=measured_contact)
