import numpy as np
import torch

from trajectory_policy.goal_policy import (
    ACTIVE_ACTION, INPUT_DIM, GoalConditionedActor, GoalConditionedValue, Standardizer,
)
from trajectory_policy.goal_ppo import goal_reward, update


def test_goal_reward_prefers_measured_tracking_and_hold_without_q_oracle():
    hand = torch.zeros(2, 11, 3)
    obj = torch.eye(4).expand(2, 4, 4).clone()
    hand_goal = hand.clone()
    obj_goal = obj.clone()
    hand_goal[1] += .2
    obj_goal[1, 0, 3] += .2
    reward, held, terms = goal_reward(
        hand, obj, hand_goal, obj_goal,
        torch.tensor([.001, .1]), torch.tensor([.1, 0.]),
        torch.tensor([False, True]), torch.tensor([False, False]),
        torch.tensor([False, False]), torch.zeros(2, 18),
    )
    assert reward[0] > reward[1]
    assert held.tolist() == [True, False]
    assert "hand_tracking" in terms and "object_tracking" in terms


def test_goal_ppo_updates_active_actor_and_finite_value_under_joint_kl_gate():
    torch.manual_seed(9)
    actor = GoalConditionedActor(width=32)
    stats = Standardizer.fit(np.zeros((4, INPUT_DIM), np.float32))
    value = GoalConditionedValue(stats, width=32)
    features = torch.randn(4, 3, INPUT_DIM) * .001
    with torch.no_grad():
        flat = features.flatten(0, 1)
        distribution = actor.distribution(flat)
        latent = distribution.sample()
        logp = distribution.log_prob(latent)[:, list(ACTIVE_ACTION)].sum(-1)
        baseline = value(flat)
    batch = dict(features=features, latent=latent.reshape(4, 3, 18),
                 logp=logp.reshape(4, 3), mean=distribution.loc.reshape(4, 3, 18),
                 std=distribution.scale.reshape(4, 3, 18), value=baseline.reshape(4, 3),
                 reward=torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12,
                 done=torch.zeros(4, 3, dtype=torch.bool))
    before = actor.network[-1].weight.detach().clone()
    result = update(actor, value, batch, torch.zeros(3),
                    torch.optim.Adam(actor.parameters(), lr=1e-4),
                    torch.optim.Adam(value.parameters(), lr=3e-4))
    assert result["behavior_logprob_error"] < .02
    assert result["joint_action_kl"] <= .02
    assert result["actor_steps"] > 0
    assert not torch.equal(before, actor.network[-1].weight)
    assert torch.isfinite(result["advantage"]).all()

