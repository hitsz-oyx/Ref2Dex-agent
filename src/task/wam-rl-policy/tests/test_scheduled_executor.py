import numpy as np
import pytest
import torch

from wam_rl_policy.goal_scheduler import OrderedGoalScheduler
from wam_rl_policy.scheduled_executor import BCWarmStartActor, ScheduledV4ExecutorInputs


def test_bc_transfer_and_ppo_probability_replay():
    torch.manual_seed(7)
    bc = torch.nn.Sequential(torch.nn.Linear(403, 16), torch.nn.ELU(),
        torch.nn.Linear(16, 16), torch.nn.ELU(), torch.nn.Linear(16, 24))
    mean, std = torch.randn(403), torch.rand(403)+.1
    actor = BCWarmStartActor(dict(variant='p2_tau', input_dim=403, hidden=16,
        state_dict=bc.state_dict(), mean=mean, std=std))
    obs = torch.randn(8, 403)
    assert torch.equal(actor(obs), bc((obs-mean)/std))
    distribution = actor.distribution(obs)
    sampled = distribution.sample()
    old_logp = distribution.log_prob(sampled).sum(-1).detach()
    assert torch.equal(old_logp, actor.distribution(obs).log_prob(sampled).sum(-1))
    before = actor.actor[0].weight.detach().clone()
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    ratio = (actor.distribution(obs).log_prob(sampled).sum(-1)-old_logp).exp()
    loss = -(ratio*torch.arange(8.)).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert not torch.equal(before, actor.actor[0].weight)
    assert torch.equal(actor.input_mean, mean)
    assert torch.equal(actor.input_std, std)


def test_live_scheduler_plan_reaches_bc_input_and_reset():
    points = np.random.RandomState(0).randn(32, 3)*.01
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 2, 3] = .1
    scheduler = OrderedGoalScheduler(poses, [0, 2], points)

    class Runtime:
        geometry = [(None, None, points)]
        device = torch.device('cpu')
        def batch(self, objects, hands, *, goal_delta):
            self.objects, self.goal = objects, goal_delta
            return goal_delta
        def predict(self, batch):
            return batch
        def tau_for_student(self, prediction, pose):
            return np.full((8, 11, 3), .001, dtype=np.float32)
        def interaction_for_executor(self, prediction):
            return np.zeros((1, 24, 128, 7)), np.ones((1, 128), bool)

    runtime = Runtime()
    bridge = ScheduledV4ExecutorInputs(runtime, scheduler)
    objects = np.zeros((7, 7)); objects[:, 6] = 1
    hands = np.zeros((7, 11, 3)); times = np.arange(7)/60
    result = bridge.observation(objects, hands, times, np.eye(4), np.zeros(139))
    assert result['observation'].shape == (403,)
    np.testing.assert_array_equal(result['observation'][139:], .001)
    np.testing.assert_array_equal(runtime.objects[0], objects[::2])
    np.testing.assert_array_equal(runtime.goal[0], result['scheduler']['goal_delta'])
    with pytest.raises(ValueError, match='strictly increasing'):
        bridge.observation(objects, hands, times, np.eye(4), np.zeros(139))
    bridge.reset()
    bridge.observation(objects, hands, times, np.eye(4), np.zeros(139))
    with pytest.raises(ValueError, match='seven consecutive'):
        bridge.observation(objects, hands, times*2, np.eye(4), np.zeros(139))


def test_full_current_bc_ppo_initialization_preserves_mean():
    from wam_rl_policy.bc_mlp import CurrentMLP,SCHEDULED_SCHEMA,SCHEDULED_DIM
    model=CurrentMLP(SCHEDULED_DIM)
    with torch.no_grad():model.net[-1].weight.normal_(std=.02)
    actor=BCWarmStartActor(dict(schema=SCHEDULED_SCHEMA,model=model.state_dict()))
    obs=torch.randn(4,SCHEDULED_DIM)
    torch.testing.assert_close(actor(obs),model(obs),rtol=0,atol=0)
    action=actor.distribution(obs).sample()
    before=actor.distribution(obs).log_prob(action).sum(-1).detach()
    assert torch.equal(before,actor.distribution(obs).log_prob(action).sum(-1))
