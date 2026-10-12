import torch

from goal_point_wam.effect_consistency import action_from_delta


def test_action_contract_reconstructs_trajectory_and_preserves_gradients():
    start = torch.randn(2, 2, 11, 3)
    delta = torch.randn(2, 24, 2, 11, 3, requires_grad=True)
    valid = torch.ones(2, 24, 2, 11, dtype=torch.bool)
    action, mask = action_from_delta(delta, start, valid)
    assert mask.all()
    torch.testing.assert_close(action[..., :3], start[:, None] + delta)
    torch.testing.assert_close(action[..., 3:6], delta)
    torch.testing.assert_close(action[:, 0, ..., 6:], delta[:, 0])
    torch.testing.assert_close(action[:, 1:, ..., 6:], delta[:, 1:] - delta[:, :-1])
    action.square().sum().backward()
    assert torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0


def test_absent_hand_and_missing_predecessor_cannot_supply_world_action():
    delta = torch.ones(1, 24, 2, 11, 3, requires_grad=True)
    valid = torch.ones(1, 24, 2, 11, dtype=torch.bool)
    valid[:, :, 1] = False
    valid[:, 4, 0] = False
    action, mask = action_from_delta(delta, torch.zeros(1, 2, 11, 3), valid)
    assert not mask[:, :, 1].any() and action[:, :, 1].count_nonzero() == 0
    assert not mask[:, 4:6, 0].any() and action[:, 4:6, 0].count_nonzero() == 0
