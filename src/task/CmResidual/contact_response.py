"""Fixed single-step interventions and physical response measurements."""
from __future__ import annotations

import torch

HORIZONS = (1, 2, 5, 10, 20, 30)


def trigger_steps(state_after, active):
    """Offline reference schedule: one step AFTER first contact-proxy event.

    This is a diagnostic schedule derived from a frozen reference rollout,
    not a deployable detector and not proof of hand-object pairwise contact.
    """
    contact = state_after[..., 49:51].bool().all(-1) & active.bool()
    found = contact.any(0)
    ticks = contact.long().argmax(0) + 1
    ticks = torch.where(found, ticks, torch.full_like(ticks, -1))
    if not found.all():
        raise ValueError('reference has environments without a contact-proxy event')
    if int(ticks.max()) + max(HORIZONS) > len(active):
        raise ValueError('insufficient post-trigger reference')
    return ticks


def pulse_action(reference, triggers, tick, amplitude):
    """Change only wrist translation z (native unit scale); preserve input."""
    if reference.ndim != 2 or reference.shape[1] != 18:
        raise ValueError('expected normalized native 18-D actions')
    action = reference.clone()
    selected = triggers.to(action.device) == tick
    requested = action[selected, 2] + amplitude
    if (requested.abs() > 1).any():
        raise ValueError('pulse clips; intended intervention would change')
    action[selected, 2] = requested
    return action


def response_features(state):
    # World object translation and angular/linear velocity; sign-invariant
    # quaternion distance is evaluated separately rather than subtracted.
    return torch.cat((state[..., 36:39], state[..., 43:49]), -1)


def sample_response(state_before, state_after, triggers, horizon):
    """Return response relative to the state just before the intervention."""
    env = torch.arange(len(triggers))
    if (triggers < 0).any() or int((triggers + horizon - 1).max()) >= len(state_after):
        raise ValueError('response window out of bounds')
    start = state_before[triggers, env]
    end = state_after[triggers + horizon - 1, env]
    return response_features(end) - response_features(start)
