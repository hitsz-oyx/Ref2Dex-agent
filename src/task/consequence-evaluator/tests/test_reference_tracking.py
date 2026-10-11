import torch
from consequence_evaluator.reference_tracking import (
    ACTIVE, INPUT_DIM, ReferenceTracker, advantages, apply_coupling, features,
    native_action, tracking_reward)


def test_native_target_inverse_preserves_wrist_and_independent_fingers():
    q = torch.zeros(2, 18); q[:, :6] = .2
    offset = torch.zeros(18); scale = torch.ones(18); scale[3:6] = torch.pi
    command = torch.zeros(2, 18); command[:, :6] = .03
    command[:, 6:] = .2
    target = offset + scale * torch.cat((command[:, :6], (command[:, 6:] + 1) / 2), -1)
    target[:, :6] += q[:, :6]
    target = apply_coupling(target)
    result = native_action(target, q, offset, scale)
    assert torch.allclose(result[:, list(ACTIVE)], command[:, list(ACTIVE)], atol=1e-6)
    assert torch.count_nonzero(result[:, [7, 9, 11, 13, 16, 17]]) == 0


def test_reference_error_survives_live_hand_drift():
    q = torch.zeros(1, 18); hand = torch.zeros(1, 11, 3)
    obj = torch.eye(4)[None]; future = torch.zeros(1, 24, 11, 3)
    args = (q, q, hand, obj, torch.zeros(1, 6), future, q, obj, torch.zeros(1, 12))
    before = features(*args)
    drifted = list(args); drifted[2] = hand + .02; drifted[0] = q + .02
    after = features(*drifted)
    assert before.shape == (1, INPUT_DIM)
    assert not torch.equal(before, after)
    # Source future remains fixed; its value must not be recentered with hand.
    assert torch.equal(before[:, 69:861], after[:, 69:861])


def test_terminal_gae_does_not_use_new_episode_value():
    rewards = torch.tensor([[1.], [2.]])
    values = torch.zeros_like(rewards); done = torch.tensor([[True], [False]])
    adv, ret = advantages(rewards, values, done, torch.tensor([100.]))
    assert adv[0].item() == 1.
    assert ret[1].item() > 2.


def test_zero_initialized_tracker_is_nominal_and_residual_is_bounded():
    model = ReferenceTracker(); obs = torch.zeros(3, INPUT_DIM); reference = torch.zeros(3, 18)
    assert torch.count_nonzero(model.actor(obs)) == 0
    target = model.target(reference, torch.full((3, 12), 100.))
    assert torch.allclose(target[:, list(ACTIVE)], model.limits.expand(3, -1))
    assert torch.allclose(target[:, 7], target[:, 6] * 1.05)


def test_object_drop_reduces_reward_even_with_identical_relative_hand():
    obj = torch.eye(4)[None]; obj[:, 2, 3] = .8
    hand = torch.zeros(1, 11, 3) + obj[:, None, :3, 3]
    initial = torch.tensor([.5]); latent = torch.zeros(1, 12)
    held = tracking_reward(hand, obj, hand, obj, torch.tensor([True]), initial, latent)
    dropped = obj.clone(); dropped[:, 2, 3] = .5
    dropped_hand = hand - torch.tensor([0., 0., .3])
    loss = tracking_reward(dropped_hand, dropped, hand, obj, torch.tensor([False]), initial, latent)
    assert loss.item() < held.item() - .5


def test_velocity_feedforward_recovers_moving_reference_pd_force():
    from consequence_evaluator.reference_tracking import wrist_feedforward
    qref = torch.full((2, 18), .2)
    velocity = torch.full_like(qref, .3)
    q = qref - .02; dq = torch.full_like(qref, .1)
    target = wrist_feedforward(qref, velocity, torch.full((6,), .1))
    force = 200 * (target[:, :6] - q[:, :6]) - 20 * dq[:, :6]
    desired = 200 * (qref[:, :6] - q[:, :6]) + 20 * (velocity[:, :6] - dq[:, :6])
    assert torch.allclose(force, desired, atol=1e-5)
    assert torch.equal(target[:, 6:], qref[:, 6:])
    # Original position-only controller omits the desired velocity term.
    assert not torch.allclose(200 * (qref[:, :6] - q[:, :6]) - 20 * dq[:, :6], desired)


def test_reference_velocity_handles_rotation_wrap_and_episode_endpoints():
    from consequence_evaluator.reference_tracking import reference_velocity
    q = torch.zeros(4, 18)
    q[:, 0] = torch.tensor([0., .01, .03, .06])
    q[:, 3] = torch.tensor([3.13, 3.14, 3.15 - 2 * torch.pi, 3.16 - 2 * torch.pi])
    velocity = reference_velocity(q, .1)
    assert torch.allclose(velocity[:, 0], torch.tensor([.1, .15, .25, .3]), atol=1e-6)
    assert torch.allclose(velocity[:, 3], torch.full((4,), .1), atol=1e-5)
    assert torch.equal(velocity[:, 6:], torch.zeros(4, 12))
