"""Causal hand/object motion evidence, explicitly NOT contact certification."""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InteractionConfig:
    near_surface_m: float = .018
    approach_distance_m: float = .06
    history_s: float = .25
    coupling_dwell_s: float = .12
    loss_dwell_s: float = .08
    relative_speed_limit_mps: float = .04
    minimum_object_speed_mps: float = .004
    coupling_relative_ratio: float = .35
    coupling_noise_mps: float = .001

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in self.__dict__.values()):
            raise ValueError('positive finite interaction heuristic limits required')


class InteractionObserver:
    """Shadow/PW layout: palm, five bases, five tips. One instance per env.

    Stationary proximity is only a contact candidate. COUPLED requires several
    observed moving-object states with consistent object-relative hand motion.
    Actual contact/force observations can still be supplied by the caller.
    """
    def __init__(self, object_points, config=None):
        self.config = config or InteractionConfig()
        self.points = np.asarray(object_points, dtype=np.float64)
        if (self.points.ndim != 2 or self.points.shape[1] != 3
                or not np.isfinite(self.points).all() or len(self.points) < 3):
            raise ValueError('finite object surface points required')
        self.reset()

    def reset(self):
        self.history = deque()
        self.coupled = False
        self.coupling_since = self.loss_since = None
        self.lost = False

    def update(self, pose, hand, now):
        c = self.config
        if hand is None:
            # Never bridge a missing-observation interval as a motion sample.
            self.reset()
            return dict(state='UNKNOWN', coupled=False, confidence=None,
                execution_ready=None, lost_control=False, near_fingers=None,
                fingertip_gap_m=None, relative_hand_speed_mps=None, object_speed_mps=None,
                reason='no_hand_observation')
        hand = np.asarray(hand, dtype=np.float64)
        if hand.shape != (11, 3) or not np.isfinite(hand).all():
            raise ValueError('finite measured palm/bases/tips hand11 required')
        local = (hand-pose[:3, 3]) @ pose[:3, :3]
        distance = np.linalg.norm(local[6:, None]-self.points[None], axis=-1).min(-1)
        near = int(np.sum(distance <= c.near_surface_m))
        gap = float(distance.min())
        world_points = self.points @ pose[:3, :3].T + pose[:3, 3]
        if self.history and now-self.history[-1][0] > c.history_s:
            self.reset()
        self.history.append((now, local.copy(), world_points))
        while len(self.history) > 1 and now-self.history[0][0] > c.history_s:
            self.history.popleft()
        age = now-self.history[0][0]
        relative_speed = object_speed = None
        if age > .03:
            relative_speed = float(np.linalg.norm(local-self.history[0][1], axis=-1).mean()/age)
            object_speed = float(np.linalg.norm(world_points-self.history[0][2], axis=-1).mean()/age)
        moving_together = (near >= 2 and relative_speed is not None
            and relative_speed < c.relative_speed_limit_mps and object_speed > c.minimum_object_speed_mps
            and relative_speed <= c.coupling_noise_mps+c.coupling_relative_ratio*object_speed)
        consistent_hold = (self.coupled and near >= 2 and relative_speed is not None
            and relative_speed < c.relative_speed_limit_mps
            and (object_speed < c.minimum_object_speed_mps or moving_together))
        if moving_together or consistent_hold:
            self.loss_since = None
            if self.coupling_since is None:
                self.coupling_since = now
            if now-self.coupling_since >= c.coupling_dwell_s:
                self.coupled, self.lost = True, False
        else:
            self.coupling_since = None
            if self.coupled:
                if self.loss_since is None:
                    self.loss_since = now
                if now-self.loss_since >= c.loss_dwell_s:
                    self.coupled, self.lost = False, True
        if self.lost:
            state, confidence, ready, reason = 'LOST', 0., False, 'observed_coupling_lost'
        elif self.coupled:
            state, confidence, ready, reason = 'COUPLED', .9, True, 'persistent_motion_coupling'
        elif gap > c.approach_distance_m:
            state, confidence, ready, reason = 'APPROACH', 0., False, 'fingers_far_from_surface'
        elif near >= 2:
            state, confidence, ready, reason = 'CONTACT_CANDIDATE', .45, None, 'proximity_without_control_certificate'
        else:
            state, confidence, ready, reason = 'PREPARE', .2, False, 'contact_configuration_incomplete'
        return dict(state=state, coupled=self.coupled, confidence=confidence,
            execution_ready=ready, lost_control=self.lost, near_fingers=near,
            fingertip_gap_m=gap, relative_hand_speed_mps=relative_speed,
            object_speed_mps=object_speed, reason=reason, score_kind='heuristic_not_probability')
