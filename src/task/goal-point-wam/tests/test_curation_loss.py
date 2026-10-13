import importlib.util
from pathlib import Path

import torch
from torch.nn import functional as F

path = Path(__file__).resolve().parents[1] / 'src/goal_point_wam/curation.py'
spec = importlib.util.spec_from_file_location('curation_loss', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unit_weights_match_original_valid_point_objective():
    pred = torch.arange(36, dtype=torch.float32).reshape(2, 3, 2, 1, 3) / 100
    target = torch.zeros_like(pred)
    valid = torch.ones(pred.shape[:-1], dtype=torch.bool)
    valid[0, :, 1] = False
    error = F.huber_loss(pred / .05, target / .05, reduction='none').mean(-1)
    actual = module.weighted_trajectory_loss(pred, target, valid, torch.ones(2))
    torch.testing.assert_close(actual, (error * valid).sum() / valid.sum())


def test_weight_scale_invariance_and_zero_weight_removes_gradient():
    pred = torch.ones(2, 3, 2, 1, 3, requires_grad=True)
    valid = torch.ones(pred.shape[:-1], dtype=torch.bool)
    target = torch.zeros_like(pred)
    a = module.weighted_trajectory_loss(pred, target, valid, torch.tensor([0., .1]))
    b = module.weighted_trajectory_loss(pred, target, valid, torch.tensor([0., 1.]))
    torch.testing.assert_close(a, b)
    a.backward()
    assert pred.grad[0].abs().sum() == 0
    assert pred.grad[1].abs().sum() > 0
