"""CPU contracts for partial thaw admission; no real PTv3/GPU allocation."""
import pytest
import torch
from torch import nn

from trajectory_policy.wam_scene_patch import (
    extract_scene_patch, load_scene_patch, scene_remainder_hash,
)


def scene():
    result = nn.Module()
    result.backbone = nn.Module()
    result.backbone.dec = nn.Module()
    result.backbone.dec.dec0 = nn.Linear(2, 2)
    result.backbone.dec.dec1 = nn.Linear(2, 2)
    result.register_buffer('scene_mean', torch.zeros(2))
    result.backbone.dec.dec0.register_buffer('counter', torch.tensor(0))
    return result


def test_patch_only_changes_whitelisted_final_decoder():
    net = scene()
    frozen = scene_remainder_hash(net)
    patch = extract_scene_patch(net)
    patch['backbone.dec.dec0.weight'].fill_(7.)
    load_scene_patch(net, patch)
    assert torch.equal(net.backbone.dec.dec0.weight, torch.full((2, 2), 7.))
    assert scene_remainder_hash(net) == frozen
    net.scene_mean.add_(1.)
    assert scene_remainder_hash(net) != frozen


@pytest.mark.parametrize('defect', ['extra', 'missing', 'shape', 'dtype', 'nonfinite'])
def test_bad_patch_rejected_atomically(defect):
    net = scene()
    original = {k: v.clone() for k, v in net.state_dict().items()}
    patch = extract_scene_patch(net)
    key = 'backbone.dec.dec0.weight'
    patch[key].fill_(3.)
    if defect == 'extra':
        patch['backbone.dec.dec1.weight'] = torch.zeros(2, 2)
    elif defect == 'missing':
        patch.pop('backbone.dec.dec0.bias')
    elif defect == 'shape':
        patch[key] = torch.zeros(1)
    elif defect == 'dtype':
        patch[key] = patch[key].double()
    else:
        patch[key][0, 0] = float('nan')
    with pytest.raises(ValueError):
        load_scene_patch(net, patch)
    assert all(torch.equal(original[k], v) for k, v in net.state_dict().items())


def test_live_PD_loss_reaches_dec0_through_frozen_R():
    net = scene()
    head, r = nn.Linear(2, 2), nn.Linear(2, 1)
    for param in list(net.parameters()) + list(r.parameters()):
        param.requires_grad_(False)
    for param in net.backbone.dec.dec0.parameters():
        param.requires_grad_(True)
    frozen_features = net.backbone.dec.dec1(torch.ones(4, 2))
    loss = r(head(net.backbone.dec.dec0(frozen_features))).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in net.backbone.dec.dec0.parameters())
    assert all(p.grad is None for p in net.backbone.dec.dec1.parameters())
    assert all(p.grad is None for p in r.parameters())
