import copy
import math

import pytest
import torch

from wam_rl_policy.motion_reference import NoReference, bounded_delta, canonical_errors
from wam_rl_policy.ppo import Actor, Critic, apply_policy_action, update


def test_bounded_causal_goal_and_world_to_native_yaw_errors():
    goal = torch.tensor([[.3, .4, 0.], [.01, 0., 0.], [0., 0., 0.]])
    delta = bounded_delta(goal, torch.zeros_like(goal))
    torch.testing.assert_close(delta, torch.tensor([[.03, .04, 0.], [.01, 0., 0.], [0., 0., 0.]]))
    target = torch.tensor([[[[0., 1., 0.]], [[0., 2., 0.]]]])
    current = torch.zeros(1, 1, 3)
    result = canonical_errors(target, current, torch.tensor([math.pi/2]))
    torch.testing.assert_close(result, torch.tensor([[[[1., 0., 0.]], [[2., 0., 0.]]]]), atol=1e-6, rtol=0)
    # A new observation changes error, never the stored world reference.
    saved = target.clone()
    canonical_errors(target, current+1, torch.zeros(1))
    assert torch.equal(saved, target)


@pytest.mark.parametrize('minibatches', [1, 2])
def test_independent_update_has_no_reference_bias_or_regularizer(minibatches):
    torch.manual_seed(37)
    actor = Actor(action_kind='independent-gaussian')
    critic = Critic()
    obs = torch.randn(4, 4, 327)
    # Nonzero compatibility slots cannot supply the legacy additive mean.
    assert not actor.distribution(obs).loc.any()
    assert not list(NoReference().model.parameters())
    assert not NoReference().act({}, obs[0, :, :300], 1).any()
    latent = torch.ones(4, 24)*.4
    applied, intended, _ = apply_policy_action(torch.ones_like(latent), latent, 'independent-gaussian')
    assert torch.equal(applied, latent) and torch.equal(intended, latent)
    with torch.no_grad():
        dist = actor.distribution(obs)
        actions = dist.sample()
        batch = dict(observation=obs, latent=actions, mean=dist.loc, std=dist.scale,
                     logp=dist.log_prob(actions).sum(-1), value=critic(obs),
                     reward=torch.randn(4, 4), done=torch.zeros(4, 4, dtype=torch.bool), bootstrap=torch.zeros(4))
    # Even an accidental nonzero legacy coefficient cannot regularize toward BC.
    clone, value_clone = copy.deepcopy(actor), copy.deepcopy(critic)
    for a, c, weight in [(actor, critic, 0.), (clone, value_clone, 100.)]:
        result = update(a, c, batch, torch.optim.Adam(a.parameters(), lr=1e-4),
                        torch.optim.Adam(c.parameters(), lr=3e-4), bc_weight=weight,
                        epochs=2, minibatches=minibatches, shuffle_generator=torch.Generator().manual_seed(38))
        assert result['bc_regularization'] == 0 and result['actor_steps'] > 0
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, clone.state_dict()[key], rtol=0, atol=0)


def test_world_reference_covariance_under_rigid_transform():
    torch.manual_seed(39)
    current = torch.randn(2, 11, 3)
    target = torch.randn(2, 2, 11, 3)
    yaw = torch.tensor([.2, -.7])
    phi = .6
    rotation = torch.tensor([[math.cos(phi), -math.sin(phi), 0.],
                             [math.sin(phi), math.cos(phi), 0.], [0., 0., 1.]])
    shift = torch.tensor([1., 2., 3.])
    expected = canonical_errors(target, current, yaw)
    actual = canonical_errors(target @ rotation.T + shift, current @ rotation.T + shift, yaw + phi)
    torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)
