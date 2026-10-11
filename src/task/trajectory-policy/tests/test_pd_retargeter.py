import numpy as np
import torch

from trajectory_policy.pd_retargeter import (PDRetargeter, command, decode_target,
                                            encode_target, pd_from_action, windows)


def test_actual_pd_roundtrip_uses_live_q_and_preserves_coupling():
    torch.manual_seed(7)
    q = torch.randn(4, 18)
    action = torch.rand(4, 18) * 1.8 - .9
    scale = torch.linspace(.5, 2., 18)
    offset = torch.linspace(-.03, .03, 18)
    original = action.clone()
    target = pd_from_action(action, q, offset, scale)
    encoded = torch.from_numpy(encode_target(target.numpy(), q.numpy()))
    reconstructed = decode_target(encoded, q)
    intended, applied, clipped = command(reconstructed, q, offset, scale)
    assert torch.equal(action, original)  # native conversion must not corrupt labels
    assert torch.allclose(target, reconstructed, atol=3e-7)
    assert torch.allclose(pd_from_action(applied, q, offset, scale), target, atol=3e-7)
    assert not clipped.any()
    shifted = q + .1
    assert torch.allclose(decode_target(encoded, shifted)[:, :6], target[:, :6] + .1)
    assert torch.allclose(decode_target(encoded, shifted)[:, 6:], target[:, 6:])


def test_window_alignment_and_terminal_boundary_ignore_future_q():
    length, n = 30, 2
    data = {k: np.zeros((length + 1, n, d), np.float32)
            for k, d in [('q', 18), ('dq', 18), ('velocity', 6)]}
    data['obj'] = np.broadcast_to(np.eye(4, dtype=np.float32), (length + 1, n, 4, 4)).copy()
    data['hand'] = np.broadcast_to(np.arange(length + 1)[:, None, None, None],
                                    (length + 1, n, 11, 3)).astype(np.float32)
    data['pd_targets'] = np.broadcast_to((100 + np.arange(length))[:, None, None],
                                         (length, n, 18)).astype(np.float32)
    state, tau, target, ticks = windows(data, [1])
    assert len(state) == length + 1 - 24
    assert ticks[-1] == 6
    assert np.array_equal(target[:, 0], 100 + ticks)
    assert np.array_equal(tau[:, 0, 0], np.ones(len(ticks)))
    assert np.array_equal(tau[:, -1, 0], np.full(len(ticks), 24))
    data['q'][-1] = 999  # terminal measured q is not the supervision target
    assert np.array_equal(windows(data, [1])[2], target)


def test_real_pd_loss_can_update_tau_and_state_only_has_no_tau_path():
    model = PDRetargeter(16)
    torch.nn.init.normal_(model.net[-1].weight, std=.1)
    state = torch.randn(3, 87)
    tau = torch.randn(3, 24, 33, requires_grad=True)
    model(state, tau).square().mean().backward()
    assert tau.grad.abs().sum() > 0
    model.use_tau = False
    assert torch.equal(model(state, tau), model(state, tau + 500))
