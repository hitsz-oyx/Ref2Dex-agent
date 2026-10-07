"""Continuous collection mechanics, independent of Isaac Gym and Torch.

Physical event diagnostics belong in a sidecar, never in evaluator Z.
No episode outcome is converted to a local preference here.
"""
import numpy as np

from .contracts import K

PHASES = ('approach', 'contact', 'grasp', 'lift', 'hold')


def pose_matrix(states):
    """Native xyz/xyzw root-state layout, retaining its stationary world frame."""
    states = np.asarray(states)
    if states.shape[-1] < 7 or not np.isfinite(states[..., :7]).all():
        raise ValueError('invalid native object root state')
    q = states[..., 3:7]
    if not np.allclose(np.linalg.norm(q, axis=-1), 1., atol=1e-3):
        raise ValueError('native object quaternion is not unit length')
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    result = np.broadcast_to(np.eye(4), (*states.shape[:-1], 4, 4)).copy()
    result[..., :3, :3] = np.stack((
        1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*states.shape[:-1], 3, 3)
    result[..., :3, 3] = states[..., :3]
    return result.astype('float32')


def smooth_residual(rng, amplitude):
    """Smooth normalized-control residual with zero boundary values, not iid jitter."""
    if not 0 < amplitude <= .2:
        raise ValueError('normalized residual amplitude must be in (0,.2]')
    knots = rng.normal(size=(4, 18))
    positions = np.linspace(0, 1, K)
    value = np.stack([np.interp(positions, np.linspace(0, 1, 4), knots[:, j])
                      for j in range(18)], -1)
    value /= np.maximum(1., np.max(np.abs(value), axis=0, keepdims=True))
    envelope = np.sin(np.pi * positions) ** 2
    scale = np.full(18, amplitude)
    scale[:3] *= .25  # Wrist translation channels use a smaller control dose.
    # These distal channels are overwritten by native Inspire coupling.
    scale[[7, 9, 11, 13, 16, 17]] = 0
    value = value * envelope[:, None] * scale
    value[[0, -1]] = 0
    return value.astype('float32')


class Perturbations:
    """One assigned phase and at most one complete 24-step residual per episode."""
    def __init__(self, count, seed, amplitude=.08, wave=0):
        self.rng = np.random.default_rng(seed)
        self.assignment = (np.arange(count) + wave) % 6  # clean then five phases
        self.chunks = np.stack([smooth_residual(self.rng, amplitude) for _ in range(count)])
        self.started = np.full(count, -1, dtype='int64')
        self.contact_run = np.zeros(count, dtype='int64')
        self.hold_run = np.zeros(count, dtype='int64')

    def known_plan(self, tick):
        """Requested residual schedule known now, before executing this action.

        A pending state-triggered intervention has an unknown future start;
        do not invent its future schedule. Once triggered, the remaining
        schedule (including zeros after the one chunk) is immutable. Actual
        clipping/noise remains a separate post-execution diagnostic.
        """
        known = (self.assignment == 0) | (self.started >= 0)
        plans = np.zeros((len(self.assignment), K, 18), dtype='float32')
        for index in np.flatnonzero(self.started >= 0):
            age = tick-self.started[index]+np.arange(K)
            active = (age >= 0) & (age < K)
            plans[index,active] = self.chunks[index,age[active]]
        return plans,known

    def apply(self, base, height, contact, initial_height, tick, remaining, active):
        base = np.asarray(base)
        if base.shape != (len(self.assignment), 18) or not np.isfinite(base).all():
            raise ValueError('invalid expert action')
        if (np.abs(base) > 1 + 1e-6).any():
            raise ValueError('expert control outside native range')
        active = np.asarray(active, dtype=bool)
        contact = np.asarray(contact, dtype=bool)
        held = (np.asarray(height) - initial_height >= .03) & contact
        # Forces immediately after reset can still belong to the preceding wave.
        self.contact_run = np.where(active & contact & (tick >= 1), self.contact_run + 1, 0)
        self.hold_run = np.where(active & held & (tick >= 1), self.hold_run + 1, 0)
        phase = np.zeros(len(base), dtype='int64')
        phase[contact] = 1
        phase[self.contact_run >= 3] = 2
        phase[held] = 3
        phase[self.hold_run >= 5] = 4
        if tick == 0:
            phase[:] = 0
        trigger = (active & (self.assignment > 0) & (self.started < 0)
                   & (self.assignment - 1 == phase) & (tick >= 1)
                   & (np.asarray(remaining) >= K))
        self.started[trigger] = tick
        age = tick - self.started
        selected = active & (self.started >= 0) & (age >= 0) & (age < K)
        residual = np.zeros_like(base)
        ids = np.flatnonzero(selected)
        residual[ids] = self.chunks[ids, age[ids]]
        unclipped = base + residual
        action = np.clip(unclipped, -1, 1)
        return action, np.asarray(PHASES)[phase], dict(
            residual=residual, actual_residual=action-base,
            clipped=np.abs(unclipped-action) > 1e-7, perturbing=selected)


class Episode:
    """T controls and T+1 states; append the actual post-action observation."""
    def __init__(self, history, object_state, contact, max_steps=600, kinematics=None):
        self.history = [np.asarray(history, dtype='float32').copy()]
        self.poses = [pose_matrix(object_state)]
        self.contact = [bool(contact)]
        self.actions, self.phases, self.actual_residual, self.clipped = [], [], [], []
        self.max_steps = max_steps
        self.finished = False
        self.plans,self.plan_valid=[],[]
        self.kinematics = {name: [np.asarray(value, dtype='float32').copy()]
                           for name, value in (kinematics or {}).items()}

    def append(self, action, phase, history, object_state, contact, residual, clipped, done, kinematics=None,
               plan=None,plan_known=False):
        if self.finished:
            raise ValueError('cannot append a reset/second episode to a completed episode')
        if len(self.actions) >= self.max_steps:
            raise ValueError('episode exceeded fixed capture budget')
        action = np.asarray(action, dtype='float32')
        history = np.asarray(history, dtype='float32')
        if action.shape != (18,) or history.shape != self.history[0].shape or not np.isfinite(history).all():
            raise ValueError('policy observation/control contract drift')
        if not np.isfinite(action).all() or np.abs(action).max() > 1+1e-6 or phase not in PHASES:
            raise ValueError('invalid executed control/current phase')
        self.actions.append(action.copy())
        plan=np.zeros((K,18),dtype='float32') if plan is None else np.asarray(plan,dtype='float32')
        if plan.shape!=(K,18) or not np.isfinite(plan).all() or np.abs(plan).max()>.2+1e-6:
            raise ValueError('invalid requested residual plan')
        self.plans.append(plan.copy());self.plan_valid.append(bool(plan_known))
        self.phases.append(str(phase))
        self.history.append(history.copy())
        self.poses.append(pose_matrix(object_state))
        self.contact.append(bool(contact))
        self.actual_residual.append(np.asarray(residual, dtype='float32').copy())
        self.clipped.append(np.asarray(clipped, dtype=bool).copy())
        self.finished = bool(done)
        if set(kinematics or {}) != set(self.kinematics):
            raise ValueError('measured kinematics contract changed')
        for name, value in (kinematics or {}).items():
            value = np.asarray(value, dtype='float32')
            if value.shape != self.kinematics[name][0].shape or not np.isfinite(value).all():
                raise ValueError('invalid measured kinematics')
            self.kinematics[name].append(value.copy())

    def arrays(self):
        if not self.finished or len(self.actions) < K:
            raise ValueError('complete episode with at least 24 actions required')
        steps = len(self.actions)
        if 'hand_keypoints' not in self.kinematics:
            raise ValueError('measured hand keypoints required; do not invent interaction future')
        return dict(history=np.asarray(self.history), action=np.asarray(self.actions),
                    residual_plan=np.asarray(self.plans),plan_known=np.asarray(self.plan_valid),
                    hand_keypoints=np.asarray(self.kinematics['hand_keypoints']),
                    object_pose=np.asarray(self.poses), timestamps=np.arange(steps+1)/30,
                    phase=np.asarray(self.phases), progress=np.full(steps+1, np.nan, dtype='float32'),
                    progress_mask=np.zeros(steps+1, dtype=bool))

    def diagnostics(self):
        """Raw physical/perturbation audit; these fields are not model inputs."""
        return dict(contact=np.asarray(self.contact), contact_valid=np.asarray([False]+[True]*len(self.actions)),
                    actual_residual=np.asarray(self.actual_residual), clipped=np.asarray(self.clipped),
                    initial_height=float(self.poses[0][2, 3]),
                    **{name: np.asarray(values) for name, values in self.kinematics.items() if name!='hand_keypoints'})
