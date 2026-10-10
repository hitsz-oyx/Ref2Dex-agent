import numpy as np
import pytest
import torch

from trajectory_policy.retargeter_data import ACTIVE, build_windows, valid_windows
from trajectory_policy.learned_retargeter import LearnedRetargeter


def recording():
    n = 8
    hand = np.broadcast_to(np.arange(n)[:, None, None, None], (n, 1, 11, 3)).astype(np.float32).copy()
    action = np.broadcast_to(np.arange(n)[:, None, None]/10, (n, 1, 18)).astype(np.float32).copy()
    return dict(q=np.zeros((n, 1, 18), np.float32), dq=np.zeros((n, 1, 18), np.float32),
        hand=hand, obj=np.broadcast_to(np.eye(4, dtype=np.float32), (n, 1, 4, 4)).copy(),
        velocity=np.zeros((n, 1, 6), np.float32), action=action, applied=action.copy(),
        episode_start=np.repeat([0, 4], 4), episode_tick=np.tile(np.arange(4), 2))


def test_actual_future_alignment_excludes_reset_and_missing_terminal():
    data = recording()
    np.testing.assert_array_equal(valid_windows(data, 2, 2), [0, 1, 4, 5])
    state, tau, label = build_windows(data, np.array([0, 4]), 2, 2)
    np.testing.assert_array_equal(tau[:, 0, :, 0], [[1, 2], [1, 2]])
    np.testing.assert_allclose(label[:, 0, :, 0], [[0, .1], [.4, .5]])
    with pytest.raises(ValueError, match='cross-reset'):
        build_windows(data, np.array([3]), 2, 2)
    # Future object/q cannot change the input of the current window.
    data['obj'][1:3, :, :3, 3] += 100
    data['q'][1:3] += 200
    changed = build_windows(data, np.array([0, 4]), 2, 2)
    np.testing.assert_array_equal(changed[0], state)
    np.testing.assert_array_equal(changed[1], tau)


def test_action_loss_can_reach_tau_while_state_only_cannot_and_native_limits_hold():
    torch.manual_seed(11)
    model = LearnedRetargeter(16).eval()
    # Move away from the intentionally constant initialization, as training does.
    torch.nn.init.normal_(model.head.weight, std=.03)
    state = torch.randn(2, 87)
    tau = torch.randn(2, 24, 33, requires_grad=True)
    model(state, tau).square().mean().backward()
    assert tau.grad.abs().sum() > 0
    with torch.no_grad():
        assert torch.equal(model(state, tau, False), model(state, tau*100, False))
        model.action_mean.fill_(3)
        native = model.native_action(state, tau)
    assert native.shape == (2, 8, 18)
    assert (native.abs() <= 1).all()
    passive = sorted(set(range(18))-set(ACTIVE))
    assert (native[..., passive] == 0).all()
