"""Ordered SE(3) path goals from measured state; no WAM or policy dependency.

The supplied path is a known instruction, not a prediction of live execution.
Object-only paths do not specify grasp readiness, release, or mandatory dwell.
Callers may supply readiness/failure observations and mandatory path indices.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from .interaction_state import InteractionObserver


def _pose(value):
    value = np.asarray(value, dtype=np.float64)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5)):
        raise ValueError('finite proper SE(3) pose required')
    return value.copy()


def _poses(values):
    """Validate a whole reference with the same per-pose tolerances as _pose."""
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 3 or values.shape[1:] != (4, 4) or not len(values)
            or not np.isfinite(values).all()
            or not np.allclose(values[:, 3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(values[:, :3, :3].transpose(0, 2, 1) @ values[:, :3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(values[:, :3, :3]), 1, atol=1e-5).all()):
        raise ValueError('finite proper SE(3) pose required')
    return values.copy()


def transform_points(pose, points):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def interpolate_pose(first, last, fraction):
    """Linear translation and shortest-arc SO(3), including pi rotations."""
    fraction = float(fraction)
    if not 0 <= fraction <= 1:
        raise ValueError('interpolation fraction outside [0,1]')
    result = np.eye(4)
    delta = Rotation.from_matrix(first[:3, :3].T @ last[:3, :3]).as_rotvec()
    result[:3, :3] = first[:3, :3] @ Rotation.from_rotvec(delta * fraction).as_matrix()
    result[:3, 3] = (1-fraction)*first[:3, 3] + fraction*last[:3, 3]
    return result


def rigid_goal_delta(current_pose, goal_pose, object_points):
    """Same-ID point displacement in the CURRENT object's fixed anchor frame."""
    current, goal = _pose(current_pose), _pose(goal_pose)
    points = np.asarray(object_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('finite Nx3 object points required')
    rotation = current[:3, :3].T @ goal[:3, :3]
    translation = current[:3, :3].T @ (goal[:3, 3]-current[:3, 3])
    return points @ rotation.T + translation - points


@dataclass(frozen=True)
class SchedulerConfig:
    horizon_s: float = .8
    milestone_spacing_m: float = .02
    position_tolerance_m: float = .008
    rotation_tolerance_rad: float = .12
    point_tolerance_m: float = .008
    lookahead_m: float = .025
    preparation_lookahead_m: float = .01
    recovery_lookahead_m: float = .008
    max_goal_translation_m: float = .05
    max_goal_rotation_rad: float = .35
    max_goal_point_m: float = .05
    recovery_enter_m: float = .04
    recovery_exit_m: float = .015
    final_settle_s: float = .2
    stall_timeout_s: float = 8.
    episode_timeout_s: float = 60.
    minimum_lookahead_m: float = .003
    lead_change_rate_mps: float = .04
    progress_deadband_m: float = .001
    max_observation_gap_s: float = .25
    max_crossing_translation_m: float = .08
    max_crossing_rotation_rad: float = .7

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in self.__dict__.values()):
            raise ValueError('scheduler limits must be positive and finite')
        if (self.milestone_spacing_m <= 2*self.point_tolerance_m
                or self.recovery_exit_m >= self.recovery_enter_m
                or self.preparation_lookahead_m > self.lookahead_m
                or self.recovery_lookahead_m > self.lookahead_m):
            raise ValueError('inconsistent scheduler spacing, hysteresis or lookahead')


class OrderedGoalScheduler:
    """One independent scheduler per environment; reset at episode boundaries.

    Only the next ordered milestone can be confirmed. Local projection selects
    a position WITHIN that segment and may retreat without undoing milestones.
    Readiness=None explicitly means unknown, rather than certified grasping.
    The module emits a goal, never executable robot commands.
    """
    def __init__(self, reference_poses, reference_times_s, object_points, *,
                 config=None, mandatory_indices=(), interaction_points=None, interaction_config=None,
                 goal_type='se3_pose'):
        if goal_type not in ('position_only', 'se3_pose'):
            raise ValueError('explicit position_only or se3_pose goal required')
        self.goal_type = goal_type
        self.config = config or SchedulerConfig()
        self.poses = _poses(reference_poses)
        if goal_type == 'position_only':
            self.poses[:, :3, :3] = self.poses[0, :3, :3]
        self.times = np.asarray(reference_times_s, dtype=np.float64).copy()
        self.points = np.asarray(object_points, dtype=np.float64).copy()
        if (len(self.poses) < 2 or self.times.shape != (len(self.poses),)
                or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0)
                or self.points.ndim != 2 or self.points.shape[1] != 3
                or len(self.points) < 3 or not np.isfinite(self.points).all()
                or np.linalg.matrix_rank(self.points-self.points.mean(0)) < 2):
            raise ValueError('ordered timestamps and noncollinear same-ID object points required')
        self.path_points = self.points @ self.poses[:, :3, :3].transpose(0, 2, 1) + self.poses[:, None, :3, 3]
        steps = np.linalg.norm(np.diff(self.path_points, axis=0), axis=-1).mean(-1)
        self.arc = np.r_[0., np.cumsum(steps)]
        required = set(int(i) for i in mandatory_indices)
        if any(i < 0 or i >= len(self.poses) for i in required):
            raise ValueError('mandatory reference index outside path')
        milestones = [0]
        for i in range(1, len(self.poses)):
            if i in required or self.arc[i]-self.arc[milestones[-1]] >= self.config.milestone_spacing_m:
                milestones.append(i)
        if milestones[-1] != len(self.poses)-1:
            milestones.append(len(self.poses)-1)
        self.milestones = np.asarray(milestones, dtype=np.int64)
        self.interaction = InteractionObserver(self.points if interaction_points is None else interaction_points,
                                               config=interaction_config)
        self.reset()

    def reset(self):
        self.pointer = 0
        self.start_time = self.last_time = self.last_progress_time = None
        self.settle_since = None
        self.initial_acquired = self.complete = self.recovering = False
        self.terminal_reason = None
        self.last_pose = None
        self.progress_anchor_arc = 0.
        self.best_finger_gap = float('inf')
        self.last_lead = self.config.minimum_lookahead_m
        self.interaction.reset()

    def _errors(self, actual, goal):
        position = float(np.linalg.norm(actual[:3, 3]-goal[:3, 3]))
        if self.goal_type == 'position_only':
            return position, 0., position
        angle = float(Rotation.from_matrix(actual[:3, :3].T @ goal[:3, :3]).magnitude())
        points = float(np.linalg.norm(transform_points(actual, self.points)-transform_points(goal, self.points), axis=-1).mean())
        return position, angle, points

    def _reached(self, actual, goal):
        p, r, q = self._errors(actual, goal)
        c = self.config
        return p <= c.position_tolerance_m and r <= c.rotation_tolerance_rad and q <= c.point_tolerance_m

    def _within_budget(self, actual, goal):
        p, r, q = self._errors(actual, goal)
        c = self.config
        return p <= c.max_goal_translation_m and r <= c.max_goal_rotation_rad and q <= c.max_goal_point_m

    def _at_arc(self, distance, first, last):
        if distance <= self.arc[first]:
            return self.poses[first].copy(), float(first)
        i = min(int(np.searchsorted(self.arc, distance, side='left')), last)
        i = max(i, first+1)
        width = self.arc[i]-self.arc[i-1]
        fraction = 0. if width <= 1e-12 else np.clip((distance-self.arc[i-1])/width, 0, 1)
        return interpolate_pose(self.poses[i-1], self.poses[i], fraction), float(i-1+fraction)

    def _bounded(self, actual, goal):
        if self._within_budget(actual, goal):
            return goal.copy()
        # Same 28 bisection decisions, but factor the invariant transform out
        # of the loop. Translation and shortest-arc angle scale linearly.
        delta = Rotation.from_matrix(actual[:3, :3].T @ goal[:3, :3]).as_rotvec()
        distance = float(np.linalg.norm(goal[:3, 3]-actual[:3, 3]))
        angle = float(np.linalg.norm(delta))
        actual_points = transform_points(actual, self.points)
        c = self.config
        low, high = 0., 1.
        for _ in range(28):
            middle = (low+high)/2
            within = distance*middle <= c.max_goal_translation_m and angle*middle <= c.max_goal_rotation_rad
            if within:
                rotation = actual[:3, :3] @ Rotation.from_rotvec(delta*middle).as_matrix()
                translation = (1-middle)*actual[:3, 3]+middle*goal[:3, 3]
                moved = self.points @ rotation.T+translation
                within = np.linalg.norm(actual_points-moved, axis=-1).mean() <= c.max_goal_point_m
            if within:
                low = middle
            else:
                high = middle
        return interpolate_pose(actual, goal, low)

    def _arrival_witness(self, actual, target, dt):
        """Observed arrival, or a bounded intervening SE(3) crossing witness.

        A crossing is a local interpolation assumption, not proof of an unseen
        arbitrary motion. Long observation gaps/jumps never supply this witness.
        """
        if self._reached(actual, target):
            return 1., 'observed_arrival'
        if self.last_pose is None or dt > self.config.max_observation_gap_s:
            return None, None
        translation, rotation, _ = self._errors(self.last_pose, actual)
        if (translation > self.config.max_crossing_translation_m
                or rotation > self.config.max_crossing_rotation_rad):
            return None, None
        if self.goal_type == 'position_only':
            start = self.last_pose[:3, 3]
            direction = actual[:3, 3]-start
            target_points = target[:3, 3]
        else:
            start = transform_points(self.last_pose, self.points)
            direction = transform_points(actual, self.points)-start
            target_points = transform_points(target, self.points)
        width = float(np.sum(direction**2))
        if width < 1e-12:
            return None, None
        fraction = float(np.clip(np.sum((target_points-start)*direction)/width, 0, 1))
        intermediate = interpolate_pose(self.last_pose, actual, fraction)
        return (fraction, 'bounded_crossing') if self._reached(intermediate, target) else (None, None)

    def update(self, object_pose, time_s, *, execution_ready=None, lost_control=False, hand_points=None):
        operation = self._update_iter(object_pose, time_s, execution_ready=execution_ready,
                                      lost_control=lost_control, hand_points=hand_points)
        try:
            actual, goal = next(operation)
            operation.send(self._bounded(actual, goal))
        except StopIteration as finished:
            return finished.value
        raise RuntimeError('unexpected extra scheduler request')

    def _update_iter(self, object_pose, time_s, *, execution_ready=None, lost_control=False, hand_points=None, _validated=False, _interaction_geometry=None):
        actual, now, c = object_pose.copy() if _validated else _pose(object_pose), float(time_s), self.config
        if self.goal_type == 'position_only':
            # Orientation is unconstrained: ask WAM for translation at the live
            # orientation, including projection, recovery and final settling.
            # The interaction observer still receives the actual SE(3) motion.
            self.poses[:, :3, :3] = actual[:3, :3]
            self.path_points = self.points @ actual[:3, :3].T + self.poses[:, None, :3, 3]
        if (not np.isfinite(now) or (self.last_time is not None and now <= self.last_time)
                or (execution_ready is not None and not isinstance(execution_ready, (bool, np.bool_)))
                or not isinstance(lost_control, (bool, np.bool_))):
            raise ValueError('strictly increasing live time and optional boolean readiness required')
        execution_ready = None if execution_ready is None else bool(execution_ready)
        readiness_external = execution_ready is not None
        dt = 0. if self.last_time is None else now-self.last_time
        interaction = self.interaction.update(actual, hand_points, now, _geometry=_interaction_geometry)
        # Weak proximity/motion proxies shape the requested goal; they cannot
        # veto independently observed geometric path progress. Only explicit
        # caller readiness/failure signals act as hard progress gates.
        hard_lost_control = lost_control
        lost_control = bool(lost_control or interaction['lost_control'])
        if self.start_time is None:
            self.start_time = self.last_progress_time = now
        self.last_time = now
        if not self.complete and now-self.start_time >= c.episode_timeout_s:
            self.terminal_reason = 'episode_deadline'
        if not self.initial_acquired and self._reached(actual, self.poses[0]):
            self.initial_acquired = True
            self.last_progress_time = now
        # Ordered crossing witnesses prevent missing milestones at lower sample
        # rates. Their interpolation fractions must also occur in order.
        arrival_witness = None
        milestone_witnesses = []
        if (self.terminal_reason is None and not self.complete and self.initial_acquired
                and execution_ready is not False and not hard_lost_control):
            previous_fraction = -1.
            for _ in range(8):
                pending = int(self.milestones[min(self.pointer+1, len(self.milestones)-1)])
                fraction, witness = self._arrival_witness(actual, self.poses[pending], dt)
                if fraction is None or fraction+1e-6 < previous_fraction:
                    self.settle_since = None
                    break
                if self.pointer+1 == len(self.milestones)-1:
                    # Crossing the end is insufficient: actual pose must settle.
                    if not self._reached(actual, self.poses[-1]):
                        self.settle_since = None
                        break
                    if self.settle_since is None:
                        self.settle_since = now
                    if now-self.settle_since >= c.final_settle_s:
                        self.pointer += 1
                        self.complete = True
                        self.last_progress_time = now
                        arrival_witness = 'final_pose_settled'
                        milestone_witnesses.append(dict(index=pending, fraction=1., kind=arrival_witness))
                    break
                self.pointer += 1
                self.last_progress_time = now
                arrival_witness = witness
                milestone_witnesses.append(dict(index=pending, fraction=fraction, kind=witness))
                previous_fraction = fraction
        else:
            self.settle_since = None
        # A tolerance-based confirmation can occur before the exact milestone.
        # Retain the preceding local segment so projection does not jump forward.
        first = int(self.milestones[max(self.pointer-1, 0)])
        last = int(self.milestones[min(self.pointer+1, len(self.milestones)-1)])
        if not self.initial_acquired:
            first = last = 0
        actual_points = transform_points(actual, self.points)
        error = np.linalg.norm(self.path_points[first:last+1]-actual_points[None], axis=-1).mean(-1)
        local = first+int(np.argmin(error))  # ties choose earliest, including static intervals
        tracking_error = float(error[local-first])
        if self.arc[local] > self.progress_anchor_arc+c.progress_deadband_m:
            self.progress_anchor_arc = float(self.arc[local])
            self.last_progress_time = now
        gap = interaction['fingertip_gap_m']
        if gap is not None and gap < self.best_finger_gap-c.progress_deadband_m:
            self.best_finger_gap = gap
            self.last_progress_time = now
        self.recovering = bool(lost_control or not self.initial_acquired or
            tracking_error > (c.recovery_exit_m if self.recovering else c.recovery_enter_m))
        if not self.complete and self.terminal_reason is None:
            if now-self.start_time >= c.episode_timeout_s:
                self.terminal_reason = 'episode_deadline'
            elif now-self.last_progress_time >= c.stall_timeout_s:
                self.terminal_reason = 'no_confirmed_milestone'
        if self.terminal_reason:
            mode, reason, goal, index = 'TIMEOUT', self.terminal_reason, actual.copy(), float(local)
        elif self.complete:
            mode, reason, goal, index = 'COMPLETE', 'final_pose_settled', self.poses[-1].copy(), float(len(self.poses)-1)
        else:
            if self.recovering:
                mode, reason, ahead = 'RECOVER', 'lost_control' if lost_control else 'outside_local_segment', c.recovery_lookahead_m
            elif execution_ready is False:
                mode, reason, ahead = 'PREPARE', 'execution_not_ready', c.preparation_lookahead_m
            else:
                mode, reason = 'TRACK', 'ordered_progress'
                ahead = c.lookahead_m * max(.25, 1-tracking_error/c.recovery_enter_m)
            if (mode != 'RECOVER' and not readiness_external
                    and interaction['state'] in ('APPROACH', 'PREPARE', 'CONTACT_CANDIDATE')):
                mode, reason = 'PREPARE', interaction['reason']
                confidence = interaction['confidence']
                ahead = c.minimum_lookahead_m + (c.preparation_lookahead_m-c.minimum_lookahead_m)*confidence/.45
            elif interaction['coupled'] and interaction['object_speed_mps'] is not None:
                # Short-horizon reach demand is related to observed motion,
                # while a positive floor preserves intent when motion pauses.
                reachable_scale = c.horizon_s*(1.5*interaction['object_speed_mps']+.005)
                ahead = min(ahead, max(c.minimum_lookahead_m, reachable_scale))
            if mode not in ('RECOVER',):
                change = c.lead_change_rate_mps*dt
                ahead = float(np.clip(ahead, self.last_lead-change, self.last_lead+change))
            available = max(0., float(self.arc[last]-self.arc[local]))
            self.last_lead = min(ahead, available)
            goal, index = self._at_arc(self.arc[local]+self.last_lead, local, last)
        # TIMEOUT emits an explicit hold at the live pose, not a reference goal.
        on_reference = mode != 'TIMEOUT'
        if not self._within_budget(actual, goal):
            if self._within_budget(actual, self.poses[local]):
                # A large rotation ahead is a NORMAL reason to shorten the
                # path goal. Interpolate along the reference, not a connector.
                low = float(self.arc[local])
                high = float(np.interp(index, np.arange(len(self.arc)), self.arc))
                for _ in range(28):
                    middle = (low+high)/2
                    candidate, _ = self._at_arc(middle, local, last)
                    if self._within_budget(actual, candidate):
                        low = middle
                    else:
                        high = middle
                goal, index = self._at_arc(low, local, last)
                reason += ':shortened_along_reference'
            else:
                goal = yield actual, goal
                on_reference = False
                mode = 'RECOVER' if mode not in ('TIMEOUT', 'COMPLETE') else mode
                reason += ':bounded_connector_to_reference'
        position, rotation, point_error = self._errors(actual, goal)
        denominator = max(float(self.arc[-1]), 1e-12)
        if on_reference and not self.complete:
            self.last_lead = max(0., float(np.interp(index, np.arange(len(self.arc)), self.arc)-self.arc[local]))
        self.last_pose = actual.copy()
        if _validated:
            rotation_delta=actual[:3,:3].T @ goal[:3,:3]
            translation_delta=actual[:3,:3].T @ (goal[:3,3]-actual[:3,3])
            goal_delta=self.points @ rotation_delta.T+translation_delta-self.points
        else:
            goal_delta=rigid_goal_delta(actual,goal,self.points)
        return dict(goal_pose=goal, goal_delta=goal_delta, goal_type=self.goal_type,
            horizon_s=c.horizon_s, goal_semantics='spatial_path_waypoint',
            reference_endpoint_time_s=None, progress_semantics='geometric_path_only',
            readiness_source='external' if readiness_external else 'heuristic_or_unknown',
            mode=mode, reason=reason, execution_ready=execution_ready,
            confirmed_index=int(self.milestones[self.pointer]), projected_index=local,
            target_reference_index=index, target_on_reference=on_reference,
            confirmed_progress=float(self.arc[self.milestones[self.pointer]]/denominator) if self.arc[-1]>1e-12 else float(self.complete),
            reference_error_m=tracking_error, goal_position_m=position, goal_rotation_rad=rotation,
            goal_point_m=point_error, complete=self.complete, timed_out=self.terminal_reason is not None,
            interaction=interaction, arrival_witness=arrival_witness, lookahead_m=self.last_lead,
            inferred_readiness=interaction['execution_ready'], milestone_witnesses=milestone_witnesses)


def bounded_connectors(schedulers, actual, goal):
    """Vectorize 28 identical SE(3) bisections; each environment retains its limits."""
    points = np.stack([s.points for s in schedulers])
    delta = Rotation.from_matrix(np.swapaxes(actual[:,:3,:3],1,2) @ goal[:,:3,:3]).as_rotvec()
    distance = np.linalg.norm(goal[:,:3,3]-actual[:,:3,3], axis=-1)
    angle = np.linalg.norm(delta, axis=-1)
    actual_points = points @ np.swapaxes(actual[:,:3,:3],1,2)+actual[:,None,:3,3]
    limits = np.array([[s.config.max_goal_translation_m,s.config.max_goal_rotation_rad,s.config.max_goal_point_m]
                       for s in schedulers])
    low=np.zeros(len(schedulers));high=np.ones(len(schedulers))
    for _ in range(28):
        middle=(low+high)/2
        rotation=actual[:,:3,:3] @ Rotation.from_rotvec(delta*middle[:,None]).as_matrix()
        translation=(1-middle[:,None])*actual[:,:3,3]+middle[:,None]*goal[:,:3,3]
        moved=points @ np.swapaxes(rotation,1,2)+translation[:,None]
        within=(distance*middle<=limits[:,0]) & (angle*middle<=limits[:,1])
        within &= np.linalg.norm(actual_points-moved,axis=-1).mean(-1)<=limits[:,2]
        low=np.where(within,middle,low);high=np.where(within,high,middle)
    result=np.broadcast_to(np.eye(4),(len(schedulers),4,4)).copy()
    result[:,:3,:3]=actual[:,:3,:3] @ Rotation.from_rotvec(delta*low[:,None]).as_matrix()
    result[:,:3,3]=(1-low[:,None])*actual[:,:3,3]+low[:,None]*goal[:,:3,3]
    return result


def update_schedulers(schedulers, poses, times, hands, *, fast_geometry=False):
    """Keep independent state machines; batch only pending connector computations."""
    geometry=[None]*len(schedulers)
    if fast_geometry:
        poses=np.asarray(poses,dtype=np.float64);hands=np.asarray(hands,dtype=np.float64)
        if (poses.shape!=(len(schedulers),4,4) or not np.isfinite(poses).all()
                or not np.allclose(poses[:,3,:],[0,0,0,1],atol=1e-6)
                or not np.allclose(poses[:,:3,:3].transpose(0,2,1) @ poses[:,:3,:3],np.eye(3),atol=1e-5)
                or not np.isclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-5).all()):
            raise ValueError('finite proper SE(3) pose required')
        if hands.shape!=(len(schedulers),11,3) or not np.isfinite(hands).all():
            raise ValueError('finite measured palm/bases/tips hand11 required')
        groups={}
        for i,s in enumerate(schedulers):groups.setdefault(len(s.interaction.points),[]).append(i)
        for ids in groups.values():
            points=np.stack([schedulers[i].interaction.points for i in ids]);p=poses[ids]
            local=(hands[ids]-p[:,None,:3,3]) @ p[:,:3,:3]
            distance=np.linalg.norm(local[:,6:,None]-points[:,None],axis=-1).min(-1)
            world=points @ p[:,:3,:3].transpose(0,2,1)+p[:,None,:3,3]
            for j,i in enumerate(ids):geometry[i]=(local[j],distance[j],world[j])
    results=[None]*len(schedulers);pending={}
    for i,scheduler in enumerate(schedulers):
        operation=scheduler._update_iter(poses[i],float(times[i]),hand_points=hands[i],_validated=fast_geometry,_interaction_geometry=geometry[i])
        try:
            actual,goal=next(operation)
            pending.setdefault(len(scheduler.points),[]).append((i,operation,actual,goal))
        except StopIteration as finished:results[i]=finished.value
    for group in pending.values():
        goals=bounded_connectors([schedulers[r[0]] for r in group],np.stack([r[2] for r in group]),np.stack([r[3] for r in group]))
        for (i,operation,_,_),goal in zip(group,goals):
            try:operation.send(goal)
            except StopIteration as finished:results[i]=finished.value
            else:raise RuntimeError('unexpected extra scheduler request')
    return results
