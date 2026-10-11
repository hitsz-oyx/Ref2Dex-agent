import numpy as np
import pytest
import torch

from trajectory_policy.pw_critic import local_inputs, truncated_returns, critic_input, PhysicalQ


def test_pw_frame_and_future_is_proposal_not_observed_future():
    obj = np.broadcast_to(np.eye(4), (8, 1, 4, 4)).copy()
    obj[:, 0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    obj[:, 0, :3, 3] = [1, 2, 3]
    hand = np.ones((8, 1, 11, 3))
    data = dict(obj=obj, hand=hand)
    proposal = np.full((24, 33), .002)
    oh, hh, hf = local_inputs(data, 3, 0, proposal)
    assert np.allclose(oh[-1], np.eye(4))
    assert np.allclose(hf - hh[-1], .002)
    hand[4:] = 1e6
    obj[4:, :, :3, 3] = 1e6
    actual = local_inputs(data, 3, 0, proposal)
    assert all(np.array_equal(a, b) for a, b in zip((oh, hh, hf), actual))


def test_complete_discounted_windows_reject_truncation_and_future_clamping():
    reward = np.arange(26, dtype=np.float32)[:, None]
    value = truncated_returns(reward, [0, 2], [0, 0])
    assert np.allclose(value, [np.dot(np.arange(24), .99 ** np.arange(24)),
                                np.dot(np.arange(2, 26), .99 ** np.arange(24))])
    with pytest.raises(ValueError):
        truncated_returns(reward, [3], [0])


def test_off_has_identical_size_without_access_to_physical_features():
    h, tau, z = np.zeros((2, 736)), np.zeros((2, 24, 33)), np.ones((2, 24, 12))
    a = critic_input(h, tau, z, False)
    assert a.shape == (2, 1816) and np.array_equal(a, critic_input(h, tau, z * 10, False))
    assert not np.array_equal(a, critic_input(h, tau, z, True))
    q = PhysicalQ(np.zeros(1816), np.ones(1816))
    x = torch.tensor(critic_input(h, tau, z, True), requires_grad=True)
    q(x).sum().backward()
    assert x.grad[:, 736:1528].abs().sum() > 0
