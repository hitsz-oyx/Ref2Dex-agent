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


GRASP_INTERFACE = 'grasp-intent.pose-preshape-closure-objectgoal.measured.v2'
FINGER_ACTIVE = np.asarray(ACTIVE)[6:]
FINGER_UPPER = np.array([1.55, 1.55, 1.55, 1.55, 1.10, .50])


@dataclass
class GraspIntent:
    """Robot-specific adapter output, not a future-frame execution schedule."""
    approach_pose_object: np.ndarray
    grasp_pose_object: np.ndarray
    preshape_q: np.ndarray
    closure_q: np.ndarray
    postgrasp_object_pose: np.ndarray
    mode: str
    tau_valid: bool
    position_only: bool = True
    approach_joint_angles: object = None  # IK branch, no extra task information


def grasp_intent_from_retargeter(q, palm, object_pose, object_goal, action,
                                offset, scale, *, plan=None, tau_valid=False):
    """Keep the trained R's wrist AND finger interpretation of WAM.

    R still reads the genuine live-conditioned tau. The controller receives only
    two wrist keyposes, independent finger joint targets, and the task goal.
    """
    from .tau_replay_contract import pd_targets
    target = pd_targets(np.asarray(action), q, offset, scale)
    mount = Rotation.from_euler('XYZ', q[3:6]).as_matrix().T @ palm[:3,:3]
    approach = np.eye(4)
    approach[:3,:3] = Rotation.from_euler('XYZ', target[3:6]).as_matrix() @ mount
    approach[:3,3] = palm[:3,3] + target[:3]-q[:3]
    local = np.linalg.inv(object_pose) @ approach
    grasp = local.copy() if plan is None else plan.local_poses[-1].copy()
    preshape = target[FINGER_ACTIVE].copy()
    closure = np.clip(np.maximum(preshape, q[FINGER_ACTIVE])+.015, .06, FINGER_UPPER)
    return GraspIntent(local, grasp, preshape, closure, np.asarray(object_goal).copy(),
                       'unavailable' if plan is None else plan.mode, bool(tau_valid),
                       approach_joint_angles=target[3:6].copy())


class MeasuredGraspConsumer:
    """Position -> measured contact -> object goal, without a reference clock."""
    def __init__(self, q, palm, offset, scale):
        self.offset, self.scale = np.asarray(offset), np.asarray(scale)
        self.mount = Rotation.from_euler('XYZ', q[3:6]).as_matrix().T @ palm[:3,:3]
        self.phase = 'approach'; self.last_time = None
        self.contact_since = None; self.contact_time = None
        self.close_pose = self.fingers = None
        self.last_info = None

    def step(self, *, q, dq, palm, object_pose, object_velocity, forces, feedback,
             near_fingers, fingertip_gap_m, intent, now):
        if self.last_time is not None and now <= self.last_time:
            raise ValueError('strictly increasing measured time required')
        if self.last_time is not None and now-self.last_time > 1/30+1e-8:
            self.contact_since = None
        self.last_time = float(now)
        self.mount=Rotation.from_euler('XYZ',q[3:6]).as_matrix().T @ palm[:3,:3]
        # Reuse the production observer; graded S2c stage scores are not sensors.
        contact = bool(feedback.force_valid and feedback.contact_observed)
        if contact:
            if self.contact_since is None:
                self.contact_since=float(now)
                if self.close_pose is not None:self.close_pose=palm.copy()
        else:self.contact_since=None
        candidate = object_pose @ intent.grasp_pose_object
        position_ready = bool(near_fingers >= 2 or
            (np.linalg.norm(candidate[:3,3]-palm[:3,3]) <= .025 and fingertip_gap_m <= .04))
        if self.phase == 'approach' and position_ready:
            self.phase='close'; self.close_pose=palm.copy()
            self.fingers=np.clip(np.maximum(intent.closure_q,q[FINGER_ACTIVE]),.06,FINGER_UPPER)
        if self.phase in ('lift','hold') and (not contact or feedback.grasp_lost):
            self.phase='close';self.close_pose=palm.copy()
        if self.phase == 'close' and self.contact_since is not None and now-self.contact_since >= .12-1e-10 and feedback.grasp_established:
            self.phase='lift';self.close_pose=palm.copy()
            self.fingers=np.clip(np.maximum(intent.closure_q,q[FINGER_ACTIVE]),.06,FINGER_UPPER)
            if self.contact_time is None:self.contact_time=float(now)
        if self.phase in ('approach','close'):
            desired_fingers=intent.preshape_q
        else:
            force=np.linalg.norm(forces,axis=-1)
            finger_force=np.r_[force[:4],force[4],force[4]]
            if feedback.force_valid:
                self.fingers+=np.where(finger_force<.25,.0015,np.where(finger_force>4.,-.001,0.))
            self.fingers=np.clip(self.fingers,.06,FINGER_UPPER)
            desired_fingers=self.fingers
        reaching = self.phase in ('approach','close')
        if reaching:
            # Closing begins at measured proximity; it must not freeze the wrist
            # before measured force-bearing grasp. Live R preserves the complete preshape.
            target=object_pose @ intent.approach_pose_object
            if self.phase=='close':self.close_pose=palm.copy()
            delta=target[:3,3]-palm[:3,3]
            angles=closest_wrist_angles(target[:3,:3],q[3:6],self.mount) if intent.approach_joint_angles is None else np.asarray(intent.approach_joint_angles)
            if not np.allclose(Rotation.from_euler('XYZ',angles).as_matrix()@self.mount,target[:3,:3],atol=1e-6,rtol=0):
                raise ValueError('wrist IK branch disagrees with pose target')
        else:
            target=self.close_pose.copy()
            if self.phase in ('lift','hold'):
                error=intent.postgrasp_object_pose[:3,3]-object_pose[:3,3]
                correction=np.clip(.40*error-.010*np.asarray(object_velocity)[:3]-.005*dq[:3],-.0035,.0035)
                target[:3,3]=palm[:3,3]+correction
                if not intent.position_only:
                    target[:3,:3]=intent.postgrasp_object_pose[:3,:3]@object_pose[:3,:3].T@palm[:3,:3]
                self.phase='hold' if np.linalg.norm(error)<=.015 else 'lift'
            delta=np.clip(target[:3,3]-palm[:3,3]-.006*dq[:3],-.006,.006)
            angles=closest_wrist_angles(target[:3,:3],q[3:6],self.mount)
            angles=q[3:6]+np.clip(.5*(angles-q[3:6])-.006*dq[3:6],-.08,.08)
        command=np.r_[(delta-self.offset[:3])/self.scale[:3],
                       (angles-q[3:6]-self.offset[3:6])/self.scale[3:6],
                       2*(desired_fingers-self.offset[FINGER_ACTIVE])/self.scale[FINGER_ACTIVE]-1].astype(np.float32)
        self.last_info=dict(phase=self.phase,measured_contact=contact,position_ready=position_ready,
            grasp_mode=intent.mode,plan_mode=intent.mode,tau_valid=intent.tau_valid,preshape_q=intent.preshape_q.tolist(),
            closure_q=intent.closure_q.tolist(),executed_finger_target=np.asarray(desired_fingers).tolist(),
            postgrasp_object_pose=intent.postgrasp_object_pose.tolist(),wrist_target=target.tolist(),
            object_goal_error_m=float(np.linalg.norm(intent.postgrasp_object_pose[:3,3]-object_pose[:3,3])))
        return command,self.last_info
