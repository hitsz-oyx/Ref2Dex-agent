"""Timed object SE(3) -> existing wrist-pose PD interface, independent of tau."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from .pose_consumer import closest_wrist_angles, FINGER_ACTIVE, FINGER_UPPER

SHAPES = ('straight', 'circle', 'figure8', 'helix', 'random_spline')


def contact_records(contacts):
    """Use the installed binding dtype; absent overlap is unknown, never zero."""
    result=[]
    for contact in contacts:
        fields=contact.dtype.names
        result.append(dict(body0=int(contact['body0']),body1=int(contact['body1']),
                           force=float(contact['lambda']),
                           initial_overlap=float(contact['initial_overlap']) if 'initial_overlap' in fields else None))
    return result


class ObjectTrajectory:
    def __init__(self, times, poses):
        self.times, self.poses = np.asarray(times, float), np.asarray(poses, float)
        if self.poses.shape != (len(self.times), 4, 4) or len(self.times) < 3 or np.any(np.diff(self.times) <= 0):
            raise ValueError('strictly increasing seconds and Nx4x4 object SE3 required')
        if not np.isfinite(self.poses).all() or not np.allclose(self.poses[:, 3], [0, 0, 0, 1]):
            raise ValueError('finite homogeneous poses required')
        rotations = self.poses[:, :3, :3]
        if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-7) or not np.allclose(np.linalg.det(rotations), 1):
            raise ValueError('SO3 required')
        self.position = CubicSpline(self.times, self.poses[:, :3, 3], bc_type='clamped')
        self.rotation = RotationSpline(self.times, Rotation.from_matrix(rotations))

    def evaluate(self, now):
        if not self.times[0]-1e-10 <= now <= self.times[-1]+1e-10:
            raise ValueError('no extrapolation beyond registered time law')
        now = np.clip(now, self.times[0], self.times[-1])
        pose = np.eye(4); pose[:3, 3] = self.position(now)
        pose[:3, :3] = self.rotation(now).as_matrix()
        # SciPy RotationSpline returns angular velocity in the rotating frame.
        omega_world = pose[:3, :3] @ self.rotation(now, 1)
        return pose, self.position(now, 1), omega_world


def make_trajectory(shape, radius, speed, initial, seed=4601):
    if shape not in SHAPES or radius <= 0 or speed <= 0:
        raise ValueError('registered shape and positive metric scale/speed required')
    grid = np.linspace(0., 1., 501)
    if shape == 'random_spline':
        rng = np.random.default_rng(seed)
        knots = np.r_[np.zeros((1, 3)), rng.uniform(-1, 1, (6, 3))]
        knots[:, 2] *= .3
        curve = CubicSpline(np.linspace(0, 1, len(knots)), knots, bc_type='clamped')
    def path(u):
        u = np.asarray(u); z = np.zeros_like(u); theta = 2*np.pi*u
        if shape == 'straight': return np.stack([u, z, z], -1)
        if shape == 'circle': return np.stack([np.sin(theta), 1-np.cos(theta), z], -1)
        if shape == 'figure8': return np.stack([np.sin(theta), .5*np.sin(2*theta), z], -1)
        if shape == 'helix': return np.stack([np.sin(theta), 1-np.cos(theta), .5*u], -1)
        return curve(u)
    length = radius*np.linalg.norm(np.diff(path(grid), axis=0), axis=-1).sum()
    nominal_duration = length/speed
    # End exactly on a native tick; report the tiny downward mean-speed rounding.
    duration = np.ceil(nominal_duration*30)/30
    u = np.linspace(0, 1, 121); phase = 10*u**3-15*u**4+6*u**5
    poses = np.repeat(np.eye(4)[None], len(u), axis=0)
    poses[:, :3, 3] = initial[:3, 3]+path(phase)*radius @ initial[:3, :3].T
    angle = np.deg2rad(10)*np.sin(2*np.pi*phase)
    poses[:, :3, :3] = initial[:3, :3] @ Rotation.from_rotvec(angle[:, None]*np.array([0., 0., 1.])).as_matrix()
    result = ObjectTrajectory(u*duration, poses)
    return result, dict(shape=shape,scale_m=radius,requested_mean_speed_m_s=speed,
                        duration_s=duration,arc_length_m=float(length),closed=shape in ('circle','figure8'),seed=seed)


def hand_setpoint(object_pose, velocity, omega_world, hand_object):
    """T_WH=T_WO*T_OH, including the rigid offset's angular lever-arm speed."""
    target = object_pose @ np.linalg.inv(hand_object)
    lever = target[:3, 3]-object_pose[:3, 3]
    return target, np.asarray(velocity)+np.cross(omega_world, lever), np.asarray(omega_world)


def _norm_cap(vector, maximum):
    vector = np.asarray(vector, float)
    return vector * min(1., maximum / max(float(np.linalg.norm(vector)), 1e-15))


class ObjectSteadyCorrection:
    """Bounded world-frame object-error integrals; no axis-specific tuning.

    Offsets bias the object reference, never the measured pose or scoring target.
    Conditional integration rejects growth when the existing command caps bind.
    Invalid contact or a finite joint near its inner boundary clears the offsets.
    A fresh instance belongs to exactly one TRACK episode.
    """
    position_gain = 1.0  # 1/s
    rotation_gain = .7  # 1/s
    position_limit = .040  # m, Euclidean norm
    rotation_limit = .14  # rad, rotation-vector norm
    position_rate = .015  # m/s
    rotation_rate = .08  # rad/s
    dt = 1. / 30

    def __init__(self):
        self.position = np.zeros(3)
        self.rotation = np.zeros(3)

    def clear(self):
        self.position[:] = 0.
        self.rotation[:] = 0.

    def propose(self, measured, desired):
        if not np.isfinite(np.r_[measured.ravel(), desired.ravel()]).all():
            raise ValueError('finite measured/reference object SE3 required')
        ep = desired[:3, 3] - measured[:3, 3]
        er = Rotation.from_matrix(desired[:3, :3] @ measured[:3, :3].T).as_rotvec()
        old = (self.position.copy(), self.rotation.copy())
        self.position = _norm_cap(self.position + self.dt * _norm_cap(self.position_gain * ep, self.position_rate), self.position_limit)
        self.rotation = _norm_cap(self.rotation + self.dt * _norm_cap(self.rotation_gain * er, self.rotation_rate), self.rotation_limit)
        return old, dict(position_error_world_m=ep.tolist(), rotation_error_world_rad=er.tolist())

    def restore(self, saved):
        self.position, self.rotation = (x.copy() for x in saved)

    def apply(self, desired):
        corrected = desired.copy()
        corrected[:3, 3] += self.position
        corrected[:3, :3] = Rotation.from_rotvec(self.rotation).as_matrix() @ desired[:3, :3]
        return corrected


class ObjectPoseTracker:
    """TRACK with PoseConsumer's existing gains, action units and native PD.

    No model, schema, goal inference or gain tuning. FF is the first-order
    velocity*dt increment in the native position-target interface, not torque.
    """
    def __init__(self, q, palm, offset, scale, *, held_finger_targets=None, joint_limits=None, predictive_braking=False,
                 object_steady_correction=False, correction_joint_limits=None):
        self.offset, self.scale = np.asarray(offset), np.asarray(scale)
        self.mount = Rotation.from_euler('XYZ', q[3:6]).as_matrix().T @ palm[:3,:3]
        self.fingers = np.asarray(q)[FINGER_ACTIVE].copy() if held_finger_targets is None else np.asarray(held_finger_targets,dtype=float).copy()
        if self.fingers.shape!=(6,) or not np.isfinite(self.fingers).all():
            raise ValueError('six finite actual preceding PD finger targets required')
        self.envelope=None;self.predictive_braking=bool(predictive_braking)
        if joint_limits is not None:
            from .reference_admission import JointEnvelope
            self.envelope=JointEnvelope(joint_limits,offset,scale)
        self.steady = ObjectSteadyCorrection() if object_steady_correction else None
        self.correction_envelope = self.envelope
        if self.steady is not None and self.correction_envelope is None:
            from .reference_admission import JointEnvelope
            if correction_joint_limits is None:
                raise ValueError('steady correction requires measured native joint limits')
            self.correction_envelope = JointEnvelope(correction_joint_limits, offset, scale)

    def step(self, q, dq, palm, target, velocity, omega_world, forces, force_valid, feedforward,
             *, measured_object=None, desired_object=None, hand_object=None):
        q, dq = np.asarray(q), np.asarray(dq)
        if self.envelope is not None:
            self.envelope.pose_configuration(target,q,palm,self.fingers)
        def wrist_law(pose, linear_ff, angular_ff):
            angles = closest_wrist_angles(pose[:3, :3], q[3:6], self.mount)
            translation = .45*(pose[:3,3]-palm[:3,3])-.012*dq[:3]
            angular = .40*(angles-q[3:6])-.008*dq[3:6]
            if feedforward:
                eps = 1e-5
                advanced = Rotation.from_rotvec(np.asarray(angular_ff)*eps).as_matrix() @ pose[:3,:3]
                derivative = (closest_wrist_angles(advanced, angles, self.mount)-angles)/eps
                translation += np.asarray(linear_ff)/30
                angular += derivative/30
            return translation, angular
        correction = None
        original_target = target.copy()
        if self.steady is not None:
            if any(x is None for x in (measured_object, desired_object, hand_object)):
                raise ValueError('measured object, original object reference and fixed grasp required')
            env = self.correction_envelope
            contact = bool(force_valid and (np.linalg.norm(forces, axis=-1) >= .1).sum() >= 2)
            near = env.mask & ((q <= env.lower) | (q >= env.upper))
            predicted = q + .10*dq
            approaching = env.mask & (((predicted <= env.lower) & (dq < 0)) | ((predicted >= env.upper) & (dq > 0)))
            correction = dict(enabled=True, accepted=False, reason=None)
            if not contact or near.any() or approaching.any():
                self.steady.clear()
                correction['reason'] = 'contact_invalid' if not contact else 'finite_joint_boundary' if near.any() else 'finite_joint_approaching'
            else:
                saved, residual = self.steady.propose(np.asarray(measured_object), np.asarray(desired_object))
                correction.update(residual)
                # Differentiate the original time law with the bias held fixed.
                # Integral updates are feedback, not extra velocity feedforward.
                object_velocity = np.asarray(velocity) - np.cross(omega_world, original_target[:3,3]-desired_object[:3,3])
                corrected_omega = Rotation.from_rotvec(self.steady.rotation).as_matrix() @ omega_world
                candidate, candidate_v, candidate_w = hand_setpoint(self.steady.apply(np.asarray(desired_object)), object_velocity, corrected_omega, hand_object)
                ct, ca = wrist_law(candidate, candidate_v, candidate_w)
                from .reference_admission import UnreachableReference
                try:
                    env.pose_configuration(candidate, q, palm, self.fingers)
                    # Check the ACTUAL incremental position target, before any
                    # clipping, including the retained FF and native damping.
                    pd = q.copy();pd[:3] += ct
                    if ((pd[:3] < env.lower[:3]) | (pd[:3] > env.upper[:3])).any():
                        raise UnreachableReference('corrected incremental PD outside inner bounds')
                except UnreachableReference:
                    self.steady.clear()
                    correction['reason'] = 'corrected_target_rejected'
                else:
                    units = np.r_[ct, ca]
                    raw_wrist = (units-self.offset[:6])/self.scale[:6]
                    if (np.abs(ct) > .025).any() or (np.abs(ca) > .12).any() or (np.abs(raw_wrist) > 1).any():
                        # Clear the applied bias while saturated; retain no
                        # stale integral that could keep driving a boundary.
                        self.steady.restore(saved)
                        correction['reason'] = 'command_saturated'
                    else:
                        target = candidate
                        velocity, omega_world = candidate_v, candidate_w
                        correction.update(accepted=True, reason='integrating')
            correction.update(position_bias_m=self.steady.position.tolist(), rotation_bias_rad=self.steady.rotation.tolist(),
                              position_at_limit=bool(np.linalg.norm(self.steady.position) >= self.steady.position_limit-1e-10),
                              rotation_at_limit=bool(np.linalg.norm(self.steady.rotation) >= self.steady.rotation_limit-1e-10))
        translation, angular = wrist_law(target, velocity, omega_world)
        # Existing pose consumer caps, unchanged in both FF conditions.
        translation = np.clip(translation, -.025, .025)
        angular = np.clip(angular, -.12, .12)
        if force_valid:
            force = np.linalg.norm(forces, axis=-1)
            finger_force = np.r_[force[:4], force[4], force[4]]
            self.fingers += np.where(finger_force < .25, .0015, np.where(finger_force > 4., -.001, 0.))
            self.fingers = np.clip(self.fingers, .06, FINGER_UPPER)
        finger_action = 2*(self.fingers-self.offset[FINGER_ACTIVE])/self.scale[FINGER_ACTIVE]-1
        raw = np.r_[(translation-self.offset[:3])/self.scale[:3],
                    (angular-self.offset[3:6])/self.scale[3:6],finger_action].astype(np.float32)
        admission=None
        if self.envelope is not None:
            raw,admission=self.envelope.protect(raw,q,dq,predictive_braking=self.predictive_braking)
            self.fingers=np.asarray(admission['executed_pd'])[FINGER_ACTIVE].copy()
        return raw, dict(phase='TRACK',wrist_target=target.tolist(),velocity_ff_world=np.asarray(velocity).tolist(),
                         angular_velocity_ff_world=np.asarray(omega_world).tolist(),feedforward=bool(feedforward),
                         clipped=bool(np.any(np.abs(raw)>1)),admission=admission,
                         original_wrist_target=original_target.tolist(),object_steady_correction=correction)


def errors(measured, desired):
    return float(np.linalg.norm(measured[:3,3]-desired[:3,3])), float(Rotation.from_matrix(desired[:3,:3].T @ measured[:3,:3]).magnitude())


def joint_margin(q, limits):
    q=np.asarray(q);mask=np.asarray(limits['has_limits'], bool)
    low=np.asarray(limits['lower']);high=np.asarray(limits['upper'])
    margin=np.minimum(q-low,high-q);margin[~mask]=np.inf
    return margin
