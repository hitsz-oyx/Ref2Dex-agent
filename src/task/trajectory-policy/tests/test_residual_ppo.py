import torch

from trajectory_policy.residual_ppo import (
    ALPHA, OBS_DIM, ResidualActor, ResidualValue, apply_residual, observation,
    residual_reward, update,
)
from trajectory_policy.retargeter_data import ACTIVE


def test_zero_residual_is_the_clipped_base_action():
    base = torch.zeros(2, 18)
    base[:, list(ACTIVE)] = torch.tensor([1.2, -1.4] + [0.25] * 10)
    applied = apply_residual(base, torch.zeros(2, 12))
    assert torch.equal(applied[:, list(ACTIVE)], base[:, list(ACTIVE)].clamp(-1, 1))
    passive = [i for i in range(18) if i not in ACTIVE]
    assert torch.equal(applied[:, passive], torch.zeros(2, len(passive)))


def test_residual_moves_only_the_twelve_active_coordinates():
    base = torch.zeros(1, 18)
    delta = torch.ones(1, 12)
    applied = apply_residual(base, delta)
    assert torch.allclose(applied[:, list(ACTIVE)], torch.full((1, 12), ALPHA))


def test_observation_uses_the_short_plan_and_three_object_poses():
    n = 3
    eye = torch.eye(4).expand(n, 4, 4).clone()
    value = observation(
        torch.zeros(n, 18), torch.zeros(n, 18), torch.zeros(n, 18), eye,
        torch.zeros(n, 6), torch.zeros(n, 11, 3), torch.zeros(n, 24, 33),
        eye[:, None].expand(n, 3, 4, 4).clone())
    assert value.shape == (n, OBS_DIM)


def test_reward_penalizes_a_throw_and_ignores_expert_actions():
    distance = torch.tensor([0.02, 0.20])
    previous = torch.tensor([0.03, 0.05])
    reward, held, terms = residual_reward(
        distance, previous, gap=torch.tensor([0.002, 0.20]),
        lift=torch.tensor([0.04, 0.27]), supported=torch.tensor([False, False]),
        previous_held=torch.tensor([False, False]), delta=torch.zeros(2, 12))
    assert held.tolist() == [True, False]
    assert terms['slip'][1] == -1
    assert reward[0] > reward[1]


def test_new_actor_mean_is_zero_and_one_update_replays():
    torch.manual_seed(296)
    actor, value = ResidualActor(), ResidualValue()
    obs = torch.randn(4, 2, OBS_DIM)
    distribution = actor.distribution(obs)
    delta = distribution.sample()
    batch = dict(observation=obs, delta=delta, reward=torch.zeros(4, 2),
                 value=torch.zeros(4, 2), done=torch.zeros(4, 2, dtype=torch.bool),
                 logp=distribution.log_prob(delta).sum(-1).detach(),
                 mean=distribution.loc.detach(), std=distribution.scale.detach())
    batch['done'][-1] = True
    result = update(actor, value, batch, torch.zeros(2),
                    torch.optim.Adam(actor.parameters(), lr=3e-4),
                    torch.optim.Adam(value.parameters(), lr=1e-3), epochs=1)
    assert result['behavior_logprob_error'] < 0.02
    assert result['actor_steps'] == 1
