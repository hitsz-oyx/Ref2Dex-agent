import torch

from wam_rl_policy_v2.measured_ppo import MeasuredCritic, MeasuredGaussianActor, update_measured


def _batch(dim):
    torch.manual_seed(7)
    t, n = 6, 2
    obs = torch.randn(t, n, dim)
    actor = MeasuredGaussianActor(dim)
    critic = MeasuredCritic(dim)
    with torch.no_grad():
        dist = actor.distribution(obs.flatten(0, 1))
        action = dist.loc.reshape(t, n, -1)
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
