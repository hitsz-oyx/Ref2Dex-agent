"""Measured TaskController -> TaskReward adapter used by online executors."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .task_reward import RewardFeatures, TaskReward

MEASURED_GAMMA = .96


def _pose_error(current, target):
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    translation = float(np.linalg.norm(target[:3, 3] - current[:3, 3]))
    rotation = float(Rotation.from_matrix(current[:3, :3].T @ target[:3, :3]).magnitude())
    return translation + rotation


def reward_features(pose, hand, object_points, decision, object_velocity, contact_forces):
    """Build nonnegative causal reward features from one measured snapshot."""
    pose = np.asarray(pose, dtype=np.float64)
    hand = np.asarray(hand, dtype=np.float64)
    points = np.asarray(object_points, dtype=np.float64)
    velocity = np.asarray(object_velocity, dtype=np.float64)
    contact = np.asarray(contact_forces, dtype=np.float64)
    if pose.shape != (4, 4) or hand.shape != (11, 3) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("measured pose/hand/object points have invalid shapes")
    if velocity.shape != (6,) or contact.shape != (5, 3):
        raise ValueError("measured object velocity/contact force shapes are required")
    world_points = points @ pose[:3, :3].T + pose[:3, 3]
    finger_distances = np.linalg.norm(hand[6:, None] - world_points[None], axis=-1).min(axis=1)
    palm_gap = float(np.linalg.norm(hand[0, None] - world_points, axis=-1).min())
    mean_tip_gap = float(finger_distances.mean())
    terminal_error = _pose_error(pose, decision.terminal_goal_pose)
    tracking_error = _pose_error(pose, decision.local_goal_pose)
    object_speed = float(np.linalg.norm(velocity[:3]))
    contact_force = float(np.linalg.norm(contact, axis=-1).mean())
    hand_clearance = float(finger_distances.mean())
    values = (palm_gap, mean_tip_gap, tracking_error, terminal_error, object_speed,
              contact_force, hand_clearance)
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError("finite nonnegative measured reward features required")
    return RewardFeatures(*values)


@dataclass
class MeasuredRewardTracker:
    """Causal one-step reward state for one TaskController episode."""

    gamma: float = MEASURED_GAMMA
    dt_s: float = 1 / 30
    effort_scale: float = .01

    def __post_init__(self):
        if not 0 < self.gamma <= 1 or self.dt_s <= 0 or self.effort_scale < 0:
            raise ValueError("valid reward timing and effort scale are required")
        self.reward = TaskReward()
        self.reset()

    def reset(self):
        self.reward.reset()
        self.previous_decision = None
        self.previous_features = None
        self.steps = 0

    def initialize(self, decision, features):
        if self.previous_decision is not None:
            raise ValueError("reward tracker already initialized")
        self.previous_decision, self.previous_features = decision, features

    def step(self, decision, features, action):
        if self.previous_decision is None or self.previous_features is None:
            raise ValueError("reward tracker requires an initial measured state")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("finite canonical 12D action required")
        effort = self.effort_scale * float(np.mean(np.square(action)))
        value, parts = self.reward.step(
            self.previous_decision, decision, self.previous_features, features,
            gamma=self.gamma, dt_s=self.dt_s, effort_cost=effort,
        )
        self.previous_decision, self.previous_features = decision, features
        self.steps += 1
        return value, parts
