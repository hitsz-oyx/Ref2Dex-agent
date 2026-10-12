"""Keep physical R meaning: predicted-action gradients belong to WAM only."""
from contextlib import contextmanager

import torch

from .wam_joint import POLICY_MODULES, TRAIN_MODULES, joint_losses

GRADIENT_OWNERSHIP = 'ref2dex.GT-tau-only-R.v1'


@contextmanager
def fixed_parameters(module):
    """Freeze parameters for graph construction without detaching its INPUT."""
    params = list(module.parameters())
    flags = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)


def anchored_losses(r, state, predicted_tau, true_tau, valid, true_pd):
    with fixed_parameters(r):
        predicted_pd = r(state, predicted_tau)
    geometry, action = joint_losses(predicted_tau, true_tau, valid,
                                    predicted_pd, true_pd, r.target_scale)
    # Pure student-state expert queries have no expert future: never train R on them.
    gt_pd = r(state[valid], true_tau[valid].detach())
    anchor = ((gt_pd-true_pd[valid])/r.target_scale).abs().mean()
    return geometry, action, anchor


def anchored_gradient_evidence(action, anchor, modules):
    if tuple(modules) != TRAIN_MODULES:
        raise ValueError('complete policy/R modules required')
    params = [p for net in modules.values() for p in net.parameters() if p.requires_grad]
    grads = torch.autograd.grad(action, params, retain_graph=True, allow_unused=True)
    evidence = {}
    for name, net in modules.items():
        owned = {id(p) for p in net.parameters()}
        selected = [g for p, g in zip(params, grads) if id(p) in owned and g is not None]
        if any(not torch.isfinite(g).all() for g in selected):
            raise ValueError('nonfinite predicted-action gradient')
        value = sum(float(g.detach().abs().sum()) for g in selected)
        if name in POLICY_MODULES and value <= 0:
            raise ValueError('missing WAM predicted-action gradient: '+name)
        if name == 'R' and value != 0:
            raise ValueError('predicted-action loss updates R')
        evidence[name] = value
    anchor_grads = torch.autograd.grad(anchor, params, retain_graph=True, allow_unused=True)
    r_owned = {id(p) for p in modules['R'].parameters()}
    r_sum = 0.
    for p, g in zip(params, anchor_grads):
        if g is not None:
            if not torch.isfinite(g).all():
                raise ValueError('nonfinite GT anchor gradient')
            value = float(g.detach().abs().sum())
            if id(p) not in r_owned and value != 0:
                raise ValueError('GT anchor updates WAM')
            r_sum += value
    if r_sum <= 0:
        raise ValueError('missing R GT anchor gradient')
    return dict(action_gradient_checks={k: evidence[k] for k in POLICY_MODULES},
                predicted_action_R_gradient_sum=evidence['R'], R_GT_gradient_sum=r_sum,
                gradient_ownership=GRADIENT_OWNERSHIP)
