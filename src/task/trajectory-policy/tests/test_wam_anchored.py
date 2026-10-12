import copy

import pytest
import torch
from torch import nn

from trajectory_policy.pd_retargeter import PDRetargeter
from trajectory_policy.wam_anchored import anchored_losses, anchored_gradient_evidence, GRADIENT_OWNERSHIP
from trajectory_policy.wam_joint import POLICY_MODULES, paired_manifest_admission
from test_wam_joint import fixture_pair


def test_gt_only_R_and_pure_current_WAM_gradient_ownership():
    torch.manual_seed(5)
    modules = {k: nn.Linear(33, 33) for k in POLICY_MODULES}
    r = PDRetargeter(16)
    nn.init.normal_(r.net[-1].weight, std=.02)
    modules['R'] = r.net
    value = torch.randn(3, 24, 33)
    for name in POLICY_MODULES:
        value = modules[name](value).tanh()
    value.retain_grad()
    state = torch.randn(3, 87)
    true = torch.randn_like(value)
    true[2] = float('nan')
    labels = torch.randn(3, 12)
    valid = torch.tensor([True, True, False])
    geometry, action, anchor = anchored_losses(r, state, value, true, valid, labels)
    evidence = anchored_gradient_evidence(action, anchor, modules)
    assert evidence['predicted_action_R_gradient_sum'] == 0
    assert evidence['R_GT_gradient_sum'] > 0
    assert all(p.requires_grad for p in r.parameters())
    gt_grad = torch.autograd.grad(anchor, tuple(r.parameters()), retain_graph=True)
    changed_labels = labels.clone()
    changed_labels[2] += 10000
    _, _, changed_anchor = anchored_losses(r, state, value, true, valid, changed_labels)
    changed_grad = torch.autograd.grad(changed_anchor, tuple(r.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(gt_grad, changed_grad))
    (geometry+action+anchor).backward()
    assert value.grad[2].abs().sum() > 0
    assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize('bad', [False, True])
def test_anchored_pair_requires_separate_R_anchor_evidence(bad):
    _, _, manifest, result = fixture_pair()
    manifest['gradient_ownership'] = GRADIENT_OWNERSHIP
    result.update(gradient_ownership=GRADIENT_OWNERSHIP, R_GT_gradient_sum=1.,
                  predicted_action_R_gradient_sum=1. if bad else 0.)
    result['action_gradient_checks'].pop('R')
    if bad:
        with pytest.raises(ValueError, match='gradient ownership'):
            paired_manifest_admission(manifest, result, copy.deepcopy(manifest), 'h', 'r')
    else:
        paired_manifest_admission(manifest, result, copy.deepcopy(manifest), 'h', 'r')
