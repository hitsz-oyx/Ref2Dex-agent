import torch
import inspect
import pytest

from trajectory_policy.residual_ppo import (
    ALPHA, OBS_DIM, ResidualActor, ResidualValue, apply_residual, observation,
    residual_reward, update,
    rollout_bootstrap, residual_execution_stats,
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


def test_residual_ppo_defaults_to_long_credit():
    assert inspect.signature(update).parameters['gae_lambda'].default == 1.0


def test_kl_backtracking_quarters_each_attempt_and_restores_rejected_actor():
    torch.manual_seed(296)
    actor, value = ResidualActor(), ResidualValue()
    obs = torch.randn(4, 2, OBS_DIM)
    dist = actor.distribution(obs)
    delta = dist.sample()
    batch = dict(observation=obs, delta=delta, reward=torch.arange(8.).reshape(4, 2),
                 value=torch.zeros(4, 2), done=torch.zeros(4, 2, dtype=torch.bool),
                 logp=dist.log_prob(delta).sum(-1).detach(),
                 mean=dist.loc.detach(), std=dist.scale.detach())
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    rates = []
    original_step = optimizer.step
    def step(*args, **kwargs):
        rates.append(optimizer.param_groups[0]['lr'])
        return original_step(*args, **kwargs)
    optimizer.step = step
    initial = {key: item.clone() for key, item in actor.state_dict().items()}
    update(actor, value, batch, torch.ones(2), optimizer,
           torch.optim.Adam(value.parameters(), lr=1e-3), target_kl=-1., epochs=1)
    assert rates == pytest.approx([3e-4, 7.5e-5, 1.875e-5, 4.6875e-6])
    assert all(torch.equal(item, initial[key]) for key, item in actor.state_dict().items())
    assert not optimizer.state


def test_truncation_bootstraps_post_control_observation_and_masks_only_terminal():
    from trajectory_policy.ppo import advantages
    class Value(torch.nn.Module):
        def forward(self, obs):
            return obs[:, 0]
    final = torch.zeros(2, OBS_DIM)
    final[:, 0] = torch.tensor([7., 11.])
    done = torch.tensor([False, True])
    bootstrap = rollout_bootstrap(Value(), final, done)
    assert bootstrap.tolist() == [7., 0.]
    advantage, returns = advantages(torch.tensor([[2., 3.]]), torch.tensor([[1., 1.]]),
        done[None], torch.ones(1, 2, dtype=torch.long), bootstrap, gamma=.99, lam=1.)
    assert torch.allclose(returns, torch.tensor([[2. + .99 * 7., 3.]]))
    assert torch.allclose(advantage, returns - 1.)


def test_lift_baseline_preserves_held_and_first_step_drop_penalty_at_late_phase():
    distance = torch.zeros(1)
    _, initial_held, _ = residual_reward(distance, distance, torch.tensor([.002]),
        torch.tensor([.04]), torch.tensor([False]), torch.tensor([False]), torch.zeros(1, 12))
    assert initial_held.item()
    _, held, terms = residual_reward(distance, distance, torch.tensor([.2]),
        torch.tensor([.04]), torch.tensor([False]), initial_held, torch.zeros(1, 12))
    assert not held.item()
    assert terms['lost'].item() == -.5


def test_saturation_is_reported_per_active_dof_from_the_executed_command():
    base = torch.zeros(2, 18)
    base[:, ACTIVE[0]] = 1.
    delta = torch.ones(2, 12)
    stats = residual_execution_stats(base, delta, apply_residual(base, delta))
    assert stats['saturation_by_dof'] == [1.] + [0.] * 11
    assert stats['positive_margin_by_dof'][0] == 0.
