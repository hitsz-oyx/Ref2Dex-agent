"""Geometry-conditioned closed-loop control with no measured future q/object."""
import time

import numpy as np
import torch
from torch import nn

from .contracts import HAND_LINKS
from .fixed_wrist_decoder import root_template, recover_wrist_sequence
from .reference_motion import NATIVE_DOF_NAMES
from .reference_tracking import ReferenceTracker, apply_coupling, features
from .reset_kinematics import ResetKinematics

SCHEMA = "ref2dex.tau-tracker.v1"
INPUT_DIM = 897
KEEP_COLUMNS = tuple(range(879)) + tuple(range(891, 909))
FINGERS = (6, 8, 10, 12, 14, 15)
FINGER_LIMITS = (1.6, 1.6, 1.6, 1.6, 1.15, .55)


def tau_calibration(packet):
    """Extract only hand geometry and initially observed robot configuration."""
    return (np.asarray(packet["hand_keypoints"][:, 0], np.float32).copy(),
            np.asarray(packet["dof_position"][0, 0], np.float32).copy())


def fit_geometry(hand, reset_q, urdf, device, iterations=300, deadline_s=110):
    """Frame-local coupled FK fit. Only hand geometry and reset q are inputs.

    Past wrist geometry resolves Euler branches; finger optimization uses two
    fixed starts, never subsequent measured joints, commands or object poses.
    """
    started = time.monotonic()
    hand = np.asarray(hand, np.float32)
    reset_q = np.asarray(reset_q, np.float32)
    if hand.ndim != 3 or hand.shape[1:] != (11, 3) or reset_q.shape != (18,):
        raise ValueError("hand episode and initial live q required")
    if not np.isfinite(hand).all() or not np.isfinite(reset_q).all():
        raise ValueError("finite geometric calibration required")
    count = len(hand)
    template = root_template(hand[0], reset_q)
    wrist = recover_wrist_sequence(hand[None], template, reset_q[None])[0]
    fixed = torch.as_tensor(np.tile(wrist, (2, 1)), device=device)
    target = torch.as_tensor(np.tile(hand, (2, 1, 1)), device=device)
    limits = torch.tensor(FINGER_LIMITS, device=device)
    initial = np.concatenate((np.broadcast_to(reset_q[list(FINGERS)], (count, 6)),
                              np.broadcast_to(np.asarray(FINGER_LIMITS) / 2, (count, 6))))
    variable = torch.tensor(initial, dtype=torch.float32, device=device)
    variable.clamp_(min=0); variable.copy_(torch.minimum(variable, limits))
    variable.requires_grad_(True)
    fk = ResetKinematics(urdf, NATIVE_DOF_NAMES, HAND_LINKS, device)
    root = torch.zeros(2 * count, 13, device=device); root[:, 6] = 1

    def coupled(value):
        q = fixed.clone(); q[:, list(FINGERS)] = value
        return apply_coupling(q)

    def points(value):
        q = coupled(value)
        return fk.states(q, torch.zeros_like(q), root)[:, :, :3]

    with torch.no_grad():
        best_error = (points(variable) - target).square().sum((1, 2))
        best = variable.detach().clone()
    optimizer = torch.optim.Adam([variable], lr=.025)
    for iteration in range(iterations):
        if time.monotonic() - started > deadline_s:
            raise TimeoutError("bounded tau geometry fitting deadline")
        optimizer.zero_grad()
        error = (points(variable) - target).square().sum((1, 2))
        with torch.no_grad():
            improved = error < best_error
            best[improved] = variable[improved]
            best_error = torch.minimum(best_error, error)
        error.mean().backward(); optimizer.step()
        with torch.no_grad():
            variable.clamp_(min=0); variable.copy_(torch.minimum(variable, limits))
    with torch.no_grad():
        final_error = (points(variable) - target).square().sum((1, 2))
        improved = final_error < best_error
        best[improved] = variable[improved]
        best_error = torch.minimum(best_error, final_error)
        choice = best_error.reshape(2, count).argmin(0)
        ids = choice * count + torch.arange(count, device=device)
        result = coupled(best)[ids]
        result[0] = torch.as_tensor(reset_q, device=device)
        fitted = fk.states(result, torch.zeros_like(result), root[:count])[:, :, :3]
    return dict(q=result.cpu().numpy(), target_points=hand.copy(), fitted_points=fitted.cpu().numpy(),
                reset_q=reset_q.copy(), chosen_start=choice.cpu().numpy())


def tau_features(q, dq, hand, obj, velocity, future_hand, geometric_q, previous):
    """No future-object argument; joint error comes from geometry, not labels."""
    full = features(q, dq, hand, obj, velocity, future_hand, geometric_q, obj, previous)
    return full[:, list(KEEP_COLUMNS)]


def tau_reward(hand, obj, reference_hand, pair, initial_height, latent):
    """Sustained-lift task reward and weak tau tracking, without future object.

    Palm displacement cannot specify object height or gate holding: the palm
    descends through approach/grasp while the object is lifted. Net force pair
    remains a training-only proxy, not a tactile policy input or force target.
    This holding objective does not implement a later placement/release task.
    """
    hand_error = (hand - reference_hand).square().mean((1, 2)).sqrt()
    actual_lift = obj[:, 2, 3] - initial_height
    lift = (actual_lift / .05).clamp(0, 1)
    hold = (pair & (actual_lift > .03)).float()
    reward = (.30 * torch.exp(-hand_error / .04) + .20 * lift + .80 * hold
              - .005 * torch.tanh(latent).square().mean(-1))
    if not torch.isfinite(reward).all():
        raise FloatingPointError("nonfinite tau tracking reward")
    return reward


class TauTracker(ReferenceTracker):
    def __init__(self):
        super().__init__()
        self.actor[0] = nn.Linear(INPUT_DIM, 128)
        self.critic[0] = nn.Linear(INPUT_DIM, 128)

    def warmstart(self, oracle_state):
        state = dict(oracle_state)
        for name in ("actor.0.weight", "critic.0.weight"):
            state[name] = state[name][:, list(KEEP_COLUMNS)]
        self.load_state_dict(state, strict=True)


def configure_finger_fit(policy):
    """Train only six finger output rows; preserve the learned wrist function."""
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    layer = policy.actor[-1]
    for parameter in (layer.weight, layer.bias):
        parameter.requires_grad_(True)
        parameter.register_hook(lambda grad: torch.cat((torch.zeros_like(grad[:6]), grad[6:]), 0))
    return [layer.weight, layer.bias]


def canonical_finger_targets(applied_pd, offset, scale, margin=.005):
    """Interior PD targets from actual commands, never loaded next joint labels.

    margin is a fraction of native finger range. It changes physical targets
    explicitly; no projection is added to the deployed controller/clip metric.
    """
    if not 0 <= margin < .1:
        raise ValueError('small declared native-range margin required')
    ids = list(FINGERS)
    lower = offset[ids] + scale[ids] * margin
    upper = offset[ids] + scale[ids] * (1 - margin)
    return torch.maximum(torch.minimum(applied_pd[..., ids], upper), lower)
