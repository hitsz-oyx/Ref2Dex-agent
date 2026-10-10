import numpy as np
import torch

from trajectory_policy.goal_policy import (
    ACTIVE_ACTION,
    GOAL_HORIZON,
    INPUT_DIM,
    RESIDUAL_GOAL_HORIZON,
    RESIDUAL_INPUT_DIM,
    GoalConditionedResidualPolicy,
    GoalConditionedPolicy,
    ResidualStandardizer,
    Standardizer,
    goal_features,
    residual_goal_features,
)


def _states(batch=2):
    q = torch.zeros(batch, 18)
    dq = torch.zeros_like(q)
    hand = torch.zeros(batch, 11, 3)
    obj = torch.eye(4).expand(batch, 4, 4).clone()
    obj_velocity = torch.zeros(batch, 6)
    prev = torch.zeros(batch, 18)
    hand_goal = hand[:, None].expand(batch, GOAL_HORIZON, 11, 3).clone()
    obj_goal = obj[:, None].expand(batch, GOAL_HORIZON, 4, 4).clone()
    return q, dq, hand, obj, obj_velocity, prev, hand_goal, obj_goal


def test_goal_features_are_bounded_shape_and_goal_sensitive_without_future_q():
    values = _states()
    features = goal_features(*values)
    assert features.shape == (2, INPUT_DIM)
    assert torch.isfinite(features).all()
    changed = list(values)
    changed[-2] = changed[-2].clone()
    changed[-2][:, 0, 0, 0] += .1
    changed_features = goal_features(*changed)
    assert not torch.equal(features, changed_features)
    assert torch.all(features.abs() <= 20)


def test_policy_masks_dependent_native_actions_and_standardizer_is_finite():
    torch.manual_seed(4)
    policy = GoalConditionedPolicy(width=32)
    output = policy(torch.randn(3, INPUT_DIM))
    assert output.shape == (3, 18)
    assert torch.all(output.abs() <= 1)
    passive = sorted(set(range(18)) - set(ACTIVE_ACTION))
    assert torch.equal(output[:, passive], torch.zeros(3, len(passive)))
    stats = Standardizer.fit(np.zeros((4, INPUT_DIM), np.float32))
    encoded = stats.encode(np.ones((2, INPUT_DIM), np.float32))
    assert np.isfinite(encoded).all()
    assert np.all(stats.scale > 0)


def test_residual_policy_emits_executor_latent():
    torch.manual_seed(8)
    policy = GoalConditionedResidualPolicy(width=32)
    output = policy(torch.randn(3, RESIDUAL_INPUT_DIM))
    assert output.shape == (3, 12)
    assert torch.isfinite(output).all()


def test_residual_features_use_24_step_goal_and_bounded_latent_history():
    q, dq, hand, obj, obj_velocity, _, _, _ = _states()
    previous = torch.zeros(2, 12)
    goal_hand = torch.zeros(2, RESIDUAL_GOAL_HORIZON, 11, 3)
    goal_obj = torch.eye(4).expand(2, RESIDUAL_GOAL_HORIZON, 4, 4).clone()
    features = residual_goal_features(q, dq, hand, obj, obj_velocity, previous,
                                      goal_hand, goal_obj)
    assert features.shape == (2, RESIDUAL_INPUT_DIM)
    previous[0, 0] = .5
    changed = residual_goal_features(q, dq, hand, obj, obj_velocity, previous,
                                     goal_hand, goal_obj)
    assert changed[0, 1155] == .5
    assert not torch.equal(features, changed)
    stats = ResidualStandardizer.fit(np.zeros((4, RESIDUAL_INPUT_DIM), np.float32))
    assert np.isfinite(stats.encode(np.ones((2, RESIDUAL_INPUT_DIM), np.float32))).all()
