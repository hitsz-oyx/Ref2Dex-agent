import math
import torch

from wam_rl_policy.coordination import rigid_fit, relative_hand, coordination_score
from wam_rl_policy.coordination import lifted_nearby


def test_rigid_fit_recovers_proper_pose_and_relative_hand():
    torch.manual_seed(41)
    points = torch.randn(3, 32, 3, dtype=torch.float64)*.04
    angle = .7
    r = torch.tensor([[math.cos(angle), -math.sin(angle), 0.],
                      [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]], dtype=torch.float64).expand(3, -1, -1)
    t = torch.randn(3, 3, dtype=torch.float64)
    target = points @ r.transpose(-1, -2) + t[:, None]
    actual_r, actual_t, error = rigid_fit(points, target)
    torch.testing.assert_close(actual_r, r)
    torch.testing.assert_close(actual_t, t)
    assert error.max() < 1e-10
    hand = torch.randn(3, 11, 3, dtype=torch.float64)
    torch.testing.assert_close(relative_hand(hand @ r.transpose(-1, -2)+t[:, None], actual_r, actual_t), hand)
    reflected = target.clone();reflected[:, :, 0] *= -1
    reflection_r, _, _ = rigid_fit(points, reflected)
    torch.testing.assert_close(torch.linalg.det(reflection_r), torch.ones(3, dtype=torch.float64))


def test_relative_score_ignores_rigid_transport_but_penalizes_slip():
    torch.manual_seed(42)
    relative = torch.randn(2, 11, 3)*.05
    rotation = torch.eye(3).expand(2, -1, -1)
    center = torch.tensor([[0., 0., .6], [1., 0., .7]])
    hand = relative + center[:, None]
    enabled = torch.tensor([True, False])
    score = coordination_score(relative_hand(hand, rotation, center), relative, enabled)
    torch.testing.assert_close(score, torch.tensor([1., 0.]))
    shift = torch.tensor([.1, .2, .3])
    transported = coordination_score(relative_hand(hand+shift, rotation, center+shift), relative, enabled)
    torch.testing.assert_close(score, transported)
    slipped = coordination_score(relative_hand(hand+.03, rotation, center), relative, enabled)
    assert 0 < slipped[0] < score[0] and slipped[1] == 0


def test_score_bounds_and_no_reward_for_predicted_object_success():
    current = torch.zeros(3, 11, 3)
    target = current.clone();target[1] += .03;target[2] += 10
    score = coordination_score(current, target, torch.ones(3, dtype=torch.bool))
    assert torch.isfinite(score).all() and (score >= 0).all() and (score <= 1).all()
    assert score[0] > score[1] > score[2]
    assert not coordination_score(current, target, torch.zeros(3, dtype=torch.bool)).any()


def test_following_a_dropped_object_does_not_preserve_phase_bonus():
    center = torch.tensor([[0., 0., .7]])
    hand = center[:, None].expand(-1, 11, -1).clone()
    start_z = torch.tensor([.6])
    assert lifted_nearby(hand, center, start_z).all()
    delta = torch.tensor([0., 0., -.1])
    assert not lifted_nearby(hand+delta, center+delta, start_z).any()
