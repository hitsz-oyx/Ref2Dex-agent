"""Meaningful invariants for temporal hindsight labels and differentiable actions."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/run'))
from ref12_methods import achieved_prefix, action_from_delta, target_roi_weights


def test_action_matches_xyz_displacement_step_contract_and_gradient():
    delta = (torch.arange(24.)[None, :, None, None, None].expand(1, 24, 2, 11, 3) + 1).clone().requires_grad_()
    start = torch.ones(1, 2, 11, 3) * 3
    valid = torch.ones(1, 24, 2, 11, dtype=torch.bool)
    action, mask = action_from_delta(delta, start, valid)
    torch.testing.assert_close(action[..., :3], start[:, None] + delta)
    torch.testing.assert_close(action[..., 3:6], delta)
    torch.testing.assert_close(action[..., 6:9], torch.ones_like(delta))
    action.square().sum().backward()
    assert delta.grad.norm() > 0 and mask.all()


def test_missing_previous_point_never_generates_fabricated_step_supervision():
    valid = torch.ones(1, 24, 2, 11, dtype=torch.bool); valid[:, 5] = False
    action, mask = action_from_delta(torch.ones(1, 24, 2, 11, 3), torch.zeros(1, 2, 11, 3), valid)
    assert not mask[:, 5:7].any()
    assert not action[:, 5:7].any()
    assert mask[:, 7:].all()


def test_hindsight_goal_uses_matching_achieved_prefix_and_keeps_world_labels_separate():
    effect = torch.eye(4).expand(3, 1, 24, 4, 4).clone()
    effect[:, :, :, 0, 3] = torch.arange(1., 25.)
    batch = dict(effect=effect, goal_object_slot=torch.zeros(3, dtype=torch.long),
                 start_object_points=torch.zeros(3, 32, 3), goal_object_points=torch.zeros(3, 32, 3),
                 trajectory_valid=torch.ones(3, 24, 2, 11, dtype=torch.bool))
    out = achieved_prefix(batch, torch.tensor([8, 16, 24]))
    torch.testing.assert_close(out['goal_object_points'][:, 0, 0], torch.tensor([8., 16., 24.]))
    torch.testing.assert_close(out['trajectory_valid'].sum((1, 2, 3)), torch.tensor([8, 16, 24]) * 22)
    assert out['effect'] is batch['effect']
    assert batch['trajectory_valid'].all() and not batch['goal_object_points'].any()


def test_roi_weights_use_current_target_and_preserve_padding():
    weights = torch.ones(2, 3, 24, 32); weights[:, 2] = 0
    result = target_roi_weights(weights, torch.tensor([0, 1]))
    assert (result[0, 0] == 4).all() and (result[0, 1] == 1).all()
    assert (result[1, 1] == 4).all() and not result[:, 2].any()
