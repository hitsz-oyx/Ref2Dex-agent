import numpy as np
import torch

from trajectory_policy.competent_actor import CompetentActor, actor_input, shifted_plan, world_plan
from trajectory_policy.pd_retargeter import PDRetargeter


def test_world_plan_retains_query_frame_when_feedback_changes():
    rotation = np.array([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]], np.float32)
    hand = np.ones((1, 11, 3), np.float32)
    tau = np.zeros((1, 24, 33), np.float32)
    tau[:, :, ::3] = np.arange(1, 25)[None, :, None]
    plan = world_plan(tau, hand, rotation)
    assert np.allclose(plan[0, :, 0, 0], 1)
    assert np.allclose(plan[0, :, 0, 1], 1 + np.arange(1, 25))
    remaining = shifted_plan(plan, 7)
    assert np.allclose(remaining[:, 0], plan[:, 7])
    assert np.allclose(remaining[:, -1], plan[:, -1])
    assert np.array_equal(plan[:, 0, :, 1], np.full((1, 11), 2))


def test_goal_and_history_contract_and_frozen_R_gradient():
    obj = np.broadcast_to(np.eye(4, dtype=np.float32), (4, 2, 4, 4)).copy()
    hand = np.zeros((4, 2, 11, 3), np.float32)
    q = np.zeros((4, 2, 18), np.float32)
    velocity = np.zeros((4, 2, 6), np.float32)
    goal = obj[-1].copy()
    goal[:, 0, 3] = .2
    x = actor_input(obj, hand, q, q, velocity, goal)
    assert x.shape == (2, 340)
    assert np.allclose(x[:, 328], .2)
    actor = CompetentActor(16)
    r = PDRetargeter(16)
    torch.nn.init.normal_(r.net[-1].weight, std=.1)
    for parameter in r.parameters():
        parameter.requires_grad_(False)
    r(torch.zeros(2, 87), actor(torch.from_numpy(x))).square().mean().backward()
    assert actor.net[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in r.parameters())
