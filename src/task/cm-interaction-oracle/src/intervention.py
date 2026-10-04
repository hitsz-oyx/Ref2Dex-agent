"""Randomized feedback-residual intervention contract; no simulator branching."""
from __future__ import annotations

import torch

ARM_NAMES = ("zero", "wrist_x_plus", "wrist_x_minus", "wrist_z_plus",
             "wrist_z_minus", "finger_plus", "finger_minus")
FINGER_INDICES = (6, 8, 10, 12, 15)
CHUNK = 4
WINDOW = 32
HISTORY = 10


def residuals(device="cpu"):
    delta = torch.zeros(7, 18, device=device)
    delta[1, 0], delta[2, 0] = .01, -.01
    delta[3, 2], delta[4, 2] = .01, -.01
    delta[5, list(FINGER_INDICES)] = .1
    delta[6, list(FINGER_INDICES)] = -.1
    return delta


def all_arms_have_headroom(base, delta):
    """Eligibility cannot depend on the arm subsequently drawn."""
    candidates = base[:, None] + delta[None]
    return ((candidates >= -1) & (candidates <= 1)).all(-1).all(-1)


def update_predecision_hold(previous_steps, height, rest_height, pair_proxy):
    """Current-state-only early-hold screen, independent of future assignment."""
    held = (height-rest_height >= .03) & pair_proxy.bool()
    return torch.where(held, previous_steps+1, torch.zeros_like(previous_steps))


def physical_targets(before, after):
    """E12 / I14 at step8, expressed in world axes (fixed airplane task).

    Physical layout: object13, measured hand poses35, hand forces15,
    object force3, body-to-object surface distances5, pair-proxy1.
    I is explicitly net-force/proximity, NOT certified paired contact/slip.
    """
    q0, q1 = before[:, 3:7], after[:, 3:7]
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    # q1 * conjugate(q0), xyzw; choose short rotation representative.
    xyz = q0[:, 3:] * q1[:, :3] - q1[:, 3:] * q0[:, :3] - torch.cross(q1[:, :3], q0[:, :3], dim=-1)
    w = (q0 * q1).sum(-1, keepdim=True)
    sign = torch.where(w < 0, -1., 1.)
    xyz, w = xyz * sign, w * sign
    norm = xyz.norm(dim=-1, keepdim=True)
    rot = xyz * (2 * torch.atan2(norm, w) / norm.clamp_min(1e-8))
    effect = torch.cat((after[:, :3] - before[:, :3], rot,
                        after[:, 7:13] - before[:, 7:13]), -1)
    hand_force = after[:, 48:63].reshape(-1, 5, 3).norm(dim=-1)
    object_force = after[:, 63:66].abs()
    interaction = torch.cat((torch.log1p(hand_force), torch.log1p(object_force),
                             after[:, 66:71], after[:, 71:72]), -1)
    return torch.cat((effect, interaction), -1)


def window_outcomes(before, trajectory, rest_height):
    """Y16/Y32: dz, proxy retention, held fraction, at-risk drop.

    Drop only has meaning for states already lifted/contacted at decision;
    retain a separate at-risk mask instead of labelling all pre-lift zeroes
    successful no-drops. Fixed rest height comes from reference frame0.
    """
    z, pair = trajectory[:, :, 2], trajectory[:, :, 71] > .5
    lift = z - rest_height[:, None]
    risk = (before[:, 2] - rest_height >= .03) & (before[:, 71] > .5)
    ys = []
    for horizon in (16, 32):
        held = (lift[:, :horizon] >= .03) & pair[:, :horizon]
        lost_run = torch.zeros(z.shape[0], device=z.device, dtype=torch.long)
        dropped = torch.zeros(z.shape[0], device=z.device, dtype=torch.bool)
        for index in range(horizon):
            lost_run = torch.where(pair[:, index], 0, lost_run + 1)
            dropped |= (lift[:, index] < .02) | (lost_run >= 6)
        ys.append(torch.stack((z[:, horizon-1] - before[:, 2],
                               pair[:, :horizon].float().mean(-1),
                               held.float().mean(-1), (dropped & risk).float()), -1))
    return torch.cat(ys, -1), risk
