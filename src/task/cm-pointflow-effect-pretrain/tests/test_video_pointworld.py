"""Observed-input boundary and masked-padding gradients without CUDA kernels."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.video_pointworld import VideoPointWorldWM


class Capture(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.projection = nn.Linear(128, 128)

    def forward(self, data):
        self.last = data
        return SimpleNamespace(feat=self.projection(data['feat']))


def model():
    with patch('oakink_wm.pointworld.PointTransformerV3', Capture):
        result = VideoPointWorldWM([0.] * 18, [1.] * 18).eval()
    nn.init.normal_(result.head[-1].weight, std=.01)
    return result


def test_forward_rejects_labels_and_reconstructs_point_identity():
    m = model()
    inputs = dict(xyz=torch.zeros(1, 2, 3), features=torch.randn(1, 2, 18),
                  point_valid=torch.ones(1, 2, dtype=torch.bool))
    flow = m(inputs)
    assert flow.shape == (1, 2, 24, 3)
    assert m.backbone.last['feat'].shape == (1, 128)
    # Coincident points share sparse geometry but retain their input skip features.
    assert not torch.equal(flow[:, 0], flow[:, 1])
    with pytest.raises(ValueError, match='observed inputs'):
        m(dict(inputs, target_flow=torch.zeros_like(flow)))


def test_invalid_input_padding_cannot_poison_output_or_gradients():
    m = model()
    features = torch.randn(1, 3, 18)
    inputs = dict(xyz=torch.zeros(1, 3, 3), features=features,
                  point_valid=torch.tensor([[True, True, False]]))
    first = m(inputs).detach()
    changed = features.clone()
    changed[:, 2] = torch.nan
    changed.requires_grad_(True)
    output = m(dict(inputs, features=changed))
    torch.testing.assert_close(first[:, :2], output[:, :2], rtol=0, atol=0)
    output[:, :2].square().sum().backward()
    assert torch.isfinite(changed.grad).all()
    assert not changed.grad[:, 2].any()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
