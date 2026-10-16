import torch
import numpy as np
from ptv3_shared_wam_v2.feature_transfer import FeatureControl, readouts, train_moments


def test_zero_feature_recovers_original_raw_controller_and_blocks_feature_gradient():
    model = FeatureControl()
    x = torch.randn(2, 1442)
    with torch.no_grad():
        model.raw[-1].weight.fill_(.01)
    z = torch.zeros(2, 2304)
    assert torch.equal(model(x, z), model.raw(x))
    model(x, z).sum().backward()
    assert model.feature.weight.grad.abs().max() == 0


def test_wrist_readout_preserves_scene_goal_and_uses_observed_wrist():
    tokens = torch.randn(2, 28, 768)
    mask = torch.ones(2, 28, dtype=torch.bool)
    mask[:, 15:26] = False
    result = readouts(tokens, mask, 4)
    assert result['wrist'].shape == result['mean'].shape == (2, 2304)
    assert torch.equal(result['wrist'][:, :768], result['mean'][:, :768])
    assert torch.equal(result['wrist'][:, 768:1536], tokens[:, 4])
    assert torch.equal(result['wrist'][:, 1536:], result['mean'][:, 1536:])


def test_feature_moments_are_fit_from_supplied_seen_rows_only():
    seen = np.array([[1., 3.], [3., 7.]], np.float32)
    mean, scale = train_moments(seen)
    np.testing.assert_array_equal(mean, [2., 5.])
    np.testing.assert_array_equal(scale, [1., 2.])
