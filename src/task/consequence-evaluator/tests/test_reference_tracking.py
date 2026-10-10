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
