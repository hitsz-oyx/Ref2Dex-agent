"""Local action-consequence contracts; no simulator dependency."""
from __future__ import annotations

import torch

SCHEMA = 'ref2dex.contact_consequence.v1'
HISTORY = 10
HORIZON = 10
EXECUTION_STEPS = 2
BASE_INDEX = 4


def local_outcomes(states, contact, trigger_z, rest_z):
    """Physical short-window labels, separate from complete grasp success.

    Contact is the documented native hand/object force proxy, not identified
    pairwise contact. Dropping is defined only for an already lifted object.
    """
    if states.shape[-2:] != (HORIZON, 49) or contact.shape != states.shape[:-1]:
        raise ValueError('incomplete consequence window')
    if not torch.isfinite(states).all():
        raise ValueError('nonfinite future state')
    z = states[..., 38]
    lift = z - trigger_z[..., None]
    supported_mm = (lift.clamp_min(0) * contact.float()).mean(-1) * 1000
    contact_fraction = contact.float().mean(-1)
    lifted = trigger_z - rest_z >= .03
    lost = torch.zeros_like(trigger_z, dtype=torch.long)
    dropped = torch.zeros_like(lifted)
    for step in range(HORIZON):
        lost = torch.where(contact[..., step], 0, lost + 1)
        dropped |= lifted & ((z[..., step] - rest_z < .02) | (lost >= 6))
    return dict(supported_lift_mm=supported_mm, contact_fraction=contact_fraction,
                contact_loss=1-contact_fraction, drop=dropped, drop_eligible=lifted,
                final_lift_mm=lift[..., -1]*1000)


def prefix_errors(actual, expected):
    """Approximate pairing tolerance; never claim a hot solver clone."""
    qa, qe = actual[..., 39:43], expected[..., 39:43]
    cosine = (torch.nn.functional.normalize(qa, dim=-1) *
              torch.nn.functional.normalize(qe, dim=-1)).sum(-1).abs().clamp(0, 1)
    errors = dict(object_position_m=(actual[..., 36:39]-expected[..., 36:39]).norm(dim=-1),
                  object_rotation_rad=2*torch.acos(cosine),
                  wrist_position_m=(actual[..., :3]-expected[..., :3]).abs().amax(-1),
                  finger_position_rad=(actual[..., 3:18]-expected[..., 3:18]).abs().amax(-1),
                  velocity=(actual[..., 18:36]-expected[..., 18:36]).abs().amax(-1),
                  object_linear_velocity=(actual[..., 43:46]-expected[..., 43:46]).abs().amax(-1),
                  object_angular_velocity=(actual[..., 46:49]-expected[..., 46:49]).abs().amax(-1))
    valid = ((errors['object_position_m'] <= .001) &
             (errors['object_rotation_rad'] <= .01) &
             (errors['wrist_position_m'] <= .001) &
             (errors['finger_position_rad'] <= .01) &
             (errors['velocity'] <= .05) &
             (errors['object_linear_velocity'] <= .05) &
             (errors['object_angular_velocity'] <= .1))
    return valid, errors


def opportunity_gate(lift, contact, drop, *, paired_valid):
    """Select with repeat0, measure opportunity on unused repeat1.

    Inputs [state, six arms, two repeats]. This is a Probe screen, not a
    validation of an executable oracle or a claim about statistical power.
    """
    if lift.ndim != 3 or lift.shape[1:] != (6, 2) or len(lift) < 24:
        raise ValueError('need at least24 complete six-arm repeat panels')
    if contact.shape != lift.shape or drop.shape != lift.shape:
        raise ValueError('outcome panel shape mismatch')
    if not torch.isfinite(lift).all() or not torch.isfinite(contact).all():
        raise ValueError('nonfinite opportunity panel')
    rows = torch.arange(len(lift), device=lift.device)
    chosen = lift[:, :, 0].argmax(-1)
    noise = (lift[:, BASE_INDEX, 1]-lift[:, BASE_INDEX, 0]).abs()
    gain = lift[rows, chosen, 1]-lift[:, BASE_INDEX, 1]
    threshold = max(2., 2*float(noise.mean()))
    resolved = gain > torch.maximum(torch.full_like(noise, 2.), 2*noise)
    contact_change = contact[rows, chosen, 1]-contact[:, BASE_INDEX, 1]
    drop_change = drop[rows, chosen, 1].float()-drop[:, BASE_INDEX, 1].float()
    passed = (paired_valid and float(gain.mean()) >= threshold and
              float(resolved.float().mean()) >= .20 and
              float(contact_change.mean()) >= -.05 and float(drop_change.mean()) <= .05)
    return dict(passed=bool(passed), states=len(lift),
                mean_confirmed_uplift_mm=float(gain.mean()),
                median_confirmed_uplift_mm=float(gain.median()),
                base_repeat_absolute_noise_mm=float(noise.mean()),
                required_mean_uplift_mm=threshold,
                resolved_opportunity_fraction=float(resolved.float().mean()),
                mean_contact_change=float(contact_change.mean()),
                all_state_drop_change=float(drop_change.mean()),
                first_repeat_choices=torch.bincount(chosen, minlength=6).tolist(),
                paired_valid=bool(paired_valid),
                label='UNCLEAR' if not paired_valid else ('PROMISING' if passed else 'UNPROMISING'),
                boundary='confirmed local candidate opportunity; no learned selector or grasp claim')
