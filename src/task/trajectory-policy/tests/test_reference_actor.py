import numpy as np
import torch

from trajectory_policy.reference_actor import (INPUT_DIM, ReferenceActor, reference_from_raw,
                                              reference_goal, reference_input)


def test_authorized_reference_offsets_tail_and_field_isolation():
    raw = np.zeros((20, 410), np.float32)
    raw[:, 204] = 1
    raw[:, 245:373] = np.tile([0, 0, 0, 1], 32)
    raw[:, 198] = np.arange(20)
    raw[:, 102:198] = np.arange(20)[:, None]
    reference = reference_from_raw(raw)
    obj = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy()
    goal = reference_goal(obj, reference, np.array([0, 19]))
    assert goal.shape == (2, 408)
    assert goal[0, 0] == 1 and goal[0, 204] == 16
    assert goal[1, 0] == 19 and goal[1, 204] == 19
    raw[:, 373:] = 500  # robot-reference q is not a G feature
    assert np.array_equal(goal, reference_goal(obj, reference_from_raw(raw), np.array([0, 19])))


def test_goal_query_frame_is_live_and_PD_loss_reaches_reference_actor():
    reference = dict(hand_pos=np.ones((20, 16, 3), np.float32),
        hand_rot=np.broadcast_to(np.eye(3, dtype=np.float32), (20, 16, 3, 3)).copy(),
        obj=np.broadcast_to(np.eye(4, dtype=np.float32), (20, 4, 4)).copy())
    obj = np.broadcast_to(np.eye(4, dtype=np.float32), (4, 2, 4, 4)).copy()
    hand = np.zeros((4, 2, 11, 3), np.float32)
    q = np.zeros((4, 2, 18), np.float32)
    velocity = np.zeros((4, 2, 6), np.float32)
    x = reference_input(obj, hand, q, q, velocity, reference, [0, 0])
    assert x.shape == (2, INPUT_DIM)
    obj[-1, :, 0, 3] = .3
    xx = reference_input(obj, hand, q, q, velocity, reference, [0, 0])
    assert np.allclose(xx[:, 328], x[:, 328] - .3)
    actor = ReferenceActor(16)
    from trajectory_policy.pd_retargeter import PDRetargeter
    r = PDRetargeter(16)
    torch.nn.init.normal_(r.net[-1].weight, std=.1)
    for parameter in r.parameters():
        parameter.requires_grad_(False)
    r(torch.zeros(2, 87), actor(torch.from_numpy(x))).square().mean().backward()
    assert actor.net[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in r.parameters())
