import torch
import pytest

from wam_rl_policy_v2.measured_ppo import MeasuredCritic, MeasuredGaussianActor, update_measured


def _batch(dim):
    torch.manual_seed(7)
    t, n = 6, 2
    obs = torch.randn(t, n, dim)
    actor = MeasuredGaussianActor(dim)
    critic = MeasuredCritic(dim)
    with torch.no_grad():
        dist = actor.distribution(obs.flatten(0, 1))
        action = dist.sample().reshape(t, n, -1)
        logp = dist.log_prob(action.flatten(0, 1)).sum(-1).reshape(t, n)
        mean = dist.loc.reshape(t, n, -1)
        std = dist.scale.reshape(t, n, -1)
        value = critic(obs.flatten(0, 1)).reshape(t, n)
    return actor, critic, dict(
        observation=obs, action=action, logp=logp, mean=mean, std=std, value=value,
        reward=torch.linspace(0.01, 0.08, t * n).reshape(t, n),
        done=torch.zeros(t, n, dtype=torch.bool), bootstrap=torch.zeros(n),
    )


def test_measured_ppo_update_accepts_both_observation_contracts():
    for dim in (421, 686):
        actor, critic, batch = _batch(dim)
        result = update_measured(actor, critic, batch, torch.optim.Adam(actor.parameters(), 1e-4),
                                 torch.optim.Adam(critic.parameters(), 1e-4), epochs=1)
        assert result["behavior_logprob_error"] < 1e-4
        assert result["actor_steps"] in (0, 1)
        assert torch.isfinite(torch.tensor(result["value_loss"]))


def test_behavior_sample_has_exploration_and_a_mean_policy_gradient():
    torch.manual_seed(11)
    actor = MeasuredGaussianActor(421)
    obs = torch.randn(24, 421)
    with torch.no_grad():
        action, old_logp, mean, std = actor.sample(obs)
    assert not torch.equal(action, mean)
    torch.testing.assert_close((action - mean).std(), std.mean(), rtol=.2, atol=.01)
    advantage = action[:, 0] - action[:, 0].mean()
    ratio = (actor.distribution(obs).log_prob(action).sum(-1) - old_logp).exp()
    (-ratio * advantage).mean().backward()
    assert actor.mean.net[-1].weight.grad.abs().max() > 1e-5


def test_reward_and_ppo_share_the_same_discount():
    from inspect import signature
    from wam_rl_policy_v2.measured_reward_bridge import MeasuredRewardTracker
    assert signature(update_measured).parameters['gamma'].default == MeasuredRewardTracker().gamma


def test_updater_rejects_the_old_deterministic_collector():
    actor, critic, batch = _batch(421)
    batch['action'] = batch['mean'].clone()
    with pytest.raises(ValueError, match='sampled behavior actions'):
        update_measured(actor, critic, batch, torch.optim.Adam(actor.parameters()),
                        torch.optim.Adam(critic.parameters()))
