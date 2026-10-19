import numpy as np
import pytest

from wam_rl_policy_v2.measured_reward_bridge import MeasuredRewardTracker, reward_features
from wam_rl_policy_v2.task_runtime import TaskController
from wam_rl_policy_v2.task_spec import TaskSpec


def fixture():
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, 0, 3] = np.arange(4) * .01
    spec = TaskSpec.from_reference(poses, source_fps=30, terminal_mode="HOLD", trim_interaction=False)
    points = np.stack((
        np.linspace(-.02, .02, 32),
        np.sin(np.linspace(0, 3.1, 32)) * .01,
        np.cos(np.linspace(0, 3.1, 32)) * .01,
    ), axis=1)
    controller = TaskController(spec, points)
    pose = poses[0]
    hand = np.zeros((11, 3), np.float64)
    velocity = np.zeros(6, np.float64)
    contact = np.zeros((5, 3), np.float64)
    decision = controller.observe(pose, hand, velocity, 0., contact_forces=contact, force_valid=False)["decision"]
    features = reward_features(pose, hand, points, decision, velocity, contact)
    return controller, pose, hand, velocity, contact, decision, features


def test_reward_features_and_tracker_are_causal_and_finite():
    controller, pose, hand, velocity, contact, decision, features = fixture()
    tracker = MeasuredRewardTracker()
    tracker.initialize(decision, features)
    next_pose = pose.copy(); next_pose[0, 3] = .001
    next_decision = controller.observe(next_pose, hand, velocity, 1 / 30., contact_forces=contact, force_valid=False)["decision"]
    next_features = reward_features(next_pose, hand, np.zeros((32, 3)), next_decision, velocity, contact)
    value, parts = tracker.step(next_decision, next_features, np.zeros(12))
    assert np.isfinite(value) and set(parts) == {"shaping", "event", "sparse", "time", "effort"}
    assert tracker.steps == 1


def test_tracker_rejects_reward_after_terminal():
    controller, pose, hand, velocity, contact, decision, features = fixture()
    tracker = MeasuredRewardTracker(); tracker.initialize(decision, features)
    terminal = controller.observe(pose, hand, velocity, 1 / 30., workspace_ok=False,
                                  contact_forces=contact, force_valid=False)["decision"]
    terminal_features = reward_features(pose, hand, np.zeros((32, 3)), terminal, velocity, contact)
    tracker.step(terminal, terminal_features, np.zeros(12))
    with pytest.raises(ValueError, match="reset reward"):
        tracker.step(terminal, terminal_features, np.zeros(12))
