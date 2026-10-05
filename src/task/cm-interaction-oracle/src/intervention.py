"""Randomized feedback-residual intervention contract; no simulator branching."""
from __future__ import annotations

import torch

ARM_NAMES = ("zero", "wrist_x_plus", "wrist_x_minus", "wrist_z_plus",
             "wrist_z_minus", "finger_plus", "finger_minus")
FINGER_INDICES = (6, 8, 10, 12, 15)
INDEPENDENT_FINGER_NAMES = ("index", "middle", "pinky", "ring", "thumb_yaw", "thumb_pitch")
INDEPENDENT_FINGER_INDICES = (6, 8, 10, 12, 14, 15)
PER_FINGER_ARM_NAMES = ("zero",) + tuple(f"{name}_{sign}" for name in INDEPENDENT_FINGER_NAMES
                                         for sign in ("plus", "minus")) + ("synergy_plus", "synergy_minus")
CHUNK = 4
WINDOW = 32
HISTORY = 10


def decode_assignment(draw, duration_levels, n_arms=len(ARM_NAMES)):
    """One uniform joint arm/duration draw, after common eligibility."""
    levels = torch.as_tensor(duration_levels, device=draw.device, dtype=torch.long)
    return draw % n_arms, levels[draw // n_arms]


def decode_amplitude_assignment(draw, amplitude_levels):
    levels = torch.as_tensor(amplitude_levels, device=draw.device, dtype=torch.float32)
    return draw % len(ARM_NAMES), levels[draw // len(ARM_NAMES)]


def surface_force_projection(force, normal):
    """Geometric projection of aggregate body force, NOT paired friction/slip."""
    normal = torch.nn.functional.normalize(normal, dim=-1)
    signed_normal = (force*normal).sum(-1)
    tangent = (force-signed_normal[..., None]*normal).norm(dim=-1)
    return signed_normal, tangent


def apply_feedback_residual(base, arms, age, durations, terminal, delta, amplitudes=None):
    treated = (arms >= 0) & (age >= 0) & (age < durations) & ~terminal
    action = base.clone()
    scale = 1 if amplitudes is None else amplitudes[treated, None]
    action[treated] = (base[treated]+scale*delta[arms[treated]]).clamp(-1, 1)
    return action


def residuals(device="cpu"):
    delta = torch.zeros(7, 18, device=device)
    delta[1, 0], delta[2, 0] = .01, -.01
    delta[3, 2], delta[4, 2] = .01, -.01
    delta[5, list(FINGER_INDICES)] = .1
    delta[6, list(FINGER_INDICES)] = -.1
    return delta


def per_finger_residuals(range_fraction=.05, device="cpu"):
    """Isolate each native independent DOF; coupling is applied by native PD.

    Native finger targets have slope range/2, so delta_action=2*fraction
    gives a signed fraction of that driver's physical range. Equal fractions
    do NOT imply equal radians, mimic fractions or fingertip millimetres.
    """
    if not 0 < range_fraction <= .25:
        raise ValueError("finger range fraction must be in (0,.25]")
    delta = torch.zeros(len(PER_FINGER_ARM_NAMES), 18, device=device)
    for finger, index in enumerate(INDEPENDENT_FINGER_INDICES):
        delta[1+2*finger, index] = 2*range_fraction
        delta[2+2*finger, index] = -2*range_fraction
    delta[-2, list(FINGER_INDICES)] = 2*range_fraction
    delta[-1, list(FINGER_INDICES)] = -2*range_fraction
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


def continuation_outcomes(before, trajectory, rest_height):
    """Early-hold prognosis AFTER step8, retaining every early-failed trial.

    Six heads: retention, held, height-or-proxy-loss failure at16 and32.
    Report physical height loss and proxy loss independently. Lost-contact
    run spans the step8 boundary. Neither is certified friction/slip.
    """
    if trajectory.shape[1] != WINDOW:
        raise ValueError("expected all32 factual post-step observations")
    risk = (before[:, 2]-rest_height >= .03) & (before[:, 71] > .5)
    z, pair = trajectory[:, :, 2], trajectory[:, :, 71] > .5
    lift = z-rest_height[:, None]
    height_failure = lift < .02
    lost = torch.zeros(len(before), device=z.device, dtype=torch.long)
    loss_failure = torch.zeros_like(pair)
    for index in range(WINDOW):
        lost = torch.where(pair[:, index], 0, lost+1)
        loss_failure[:, index] = lost >= 6
    failures = (height_failure | loss_failure) & risk[:, None]
    first = torch.where(failures.any(-1), failures.long().argmax(-1)+1, -1)
    heads = []
    for endpoint in (16, 32):
        region = slice(8, endpoint)
        heads.append(torch.stack((pair[:, region].float().mean(-1),
            ((lift[:, region] >= .03) & pair[:, region]).float().mean(-1),
            failures[:, region].any(-1).float()), -1))
    details = dict(risk=risk, first_failure_step=first,
        early_failure=failures[:, :8].any(-1),
        all32_failure=failures.any(-1), height_failure32=(height_failure & risk[:, None]).any(-1),
        proxy_loss_failure32=(loss_failure & risk[:, None]).any(-1),
        late_height_failure=height_failure[:, 8:].any(-1) & risk,
        late_proxy_loss_failure=loss_failure[:, 8:].any(-1) & risk,
        short_contact_fraction=pair[:, :8].float().mean(-1))
    return torch.cat(heads, -1), details
