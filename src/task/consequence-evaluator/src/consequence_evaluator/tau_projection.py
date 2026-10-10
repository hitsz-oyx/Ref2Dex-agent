"""Batched bounded URDF projection of candidate hand trajectories."""
import time
import numpy as np
import torch

from .contracts import HAND_LINKS
from .fixed_wrist_decoder import root_template, recover_wrist_sequence
from .reference_motion import NATIVE_DOF_NAMES
from .reference_tracking import apply_coupling
from .reset_kinematics import ResetKinematics
from .tau_tracking import FINGERS, FINGER_LIMITS


def project_tau(current_hand, future_hand, current_q, urdf, device, iterations=300,
                deadline_s=480, monitor=None):
    """No future q/action/object/force labels; current q calibrates each query.

    Same two-start frame-local coupled finger fit as the existing tau tracker.
    Returns geometry only, without claiming that fitted q supplies PD preload.
    """
    current = np.asarray(current_hand, dtype='float32')
    future = np.asarray(future_hand, dtype='float32')
    reset = np.asarray(current_q, dtype='float32')
    if (current.shape != (len(future), 11, 3) or future.ndim != 4
            or future.shape[2:] != (11, 3) or reset.shape != (len(future), 18)):
        raise ValueError('query/current/future batch shapes mismatch')
    if not all(np.isfinite(v).all() for v in (current, future, reset)):
        raise ValueError('finite hand/current q required')
    started = time.monotonic(); n = len(future); length = future.shape[1]+1
    hand = np.concatenate((current[:, None], future), 1)
    wrists = []
    for i in range(n):
        template = root_template(current[i], reset[i])
        wrists.append(recover_wrist_sequence(hand[i:i+1], template, reset[i:i+1])[0])
    wrist = np.stack(wrists).reshape(-1, 18); count = len(wrist)
    fixed = torch.tensor(np.tile(wrist, (2, 1)), device=device)
    target = torch.tensor(np.tile(hand.reshape(-1, 11, 3), (2, 1, 1)), device=device)
    limits = torch.tensor(FINGER_LIMITS, device=device)
    live_fingers = np.repeat(reset[:, list(FINGERS)], length, axis=0)
    initial = np.concatenate((live_fingers, np.broadcast_to(np.asarray(FINGER_LIMITS)/2, (count, 6))))
    variable = torch.tensor(initial, dtype=torch.float32, device=device)
    variable.clamp_(min=0); variable.copy_(torch.minimum(variable, limits)); variable.requires_grad_(True)
    fk = ResetKinematics(urdf, NATIVE_DOF_NAMES, HAND_LINKS, device)
    root = torch.zeros(2*count, 13, device=device); root[:, 6] = 1

    def coupled(v):
        q = fixed.clone(); q[:, list(FINGERS)] = v
        return apply_coupling(q)

    def points(v):
        q = coupled(v)
        return fk.states(q, torch.zeros_like(q), root)[:, :, :3]

    with torch.no_grad():
        best_error = (points(variable)-target).square().sum((1, 2)); best = variable.detach().clone()
    optimizer = torch.optim.Adam([variable], lr=.025)
    for iteration in range(iterations):
        if time.monotonic()-started > deadline_s:
            raise TimeoutError('bounded candidate projection deadline')
        optimizer.zero_grad()
        error = (points(variable)-target).square().sum((1, 2))
        with torch.no_grad():
            improved = error < best_error; best[improved] = variable[improved]
            best_error = torch.minimum(best_error, error)
        error.mean().backward(); optimizer.step()
        with torch.no_grad():
            variable.clamp_(min=0); variable.copy_(torch.minimum(variable, limits))
        if monitor and (iteration == 0 or (iteration+1)%50 == 0):
            monitor(iteration+1, time.monotonic()-started)
    with torch.no_grad():
        error = (points(variable)-target).square().sum((1, 2))
        improved = error < best_error; best[improved] = variable[improved]
        best_error = torch.minimum(best_error, error)
        choice = best_error.reshape(2, count).argmin(0)
        ids = choice*count+torch.arange(count, device=device)
        q = coupled(best)[ids].reshape(n, length, 18)
        q[:, 0] = torch.tensor(reset, device=device)
        flat = q.reshape(-1, 18)
        fitted = fk.states(flat, torch.zeros_like(flat), root[:count])[:, :, :3].reshape(n, length, 11, 3)
    return dict(q=q.cpu().numpy(), points=fitted.cpu().numpy(),
                elapsed_s=time.monotonic()-started, chosen_start=choice.cpu().numpy().reshape(n, length))
