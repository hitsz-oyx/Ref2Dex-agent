"""Factual-path rolling short-Y observability, never a trajectory-switch controller."""
import numpy as np
import torch

OFFSETS = tuple(range(0, 57, 8))
DENSE_OFFSETS = tuple(range(59))
EPSILON = .02


def short_y(current_height, current_pair, height, pair, rest):
    """Exact inherited Y8 from its sufficient height/force-pair fields.

    Local loss history starts at zero and spans local step8. Every window has
    all32 future observations; current risk conditions failure heads only.
    """
    if height.shape != pair.shape or height.shape[-1] != 32:
        raise ValueError('all32 future observations required')
    if current_height.shape != height.shape[:-1] or current_pair.shape != current_height.shape or rest.shape != current_height.shape:
        raise ValueError('current/window identity mismatch')
    if not torch.isfinite(height).all() or not torch.isfinite(current_height).all() or not torch.isfinite(rest).all():
        raise ValueError('nonfinite height')
    pair = pair.bool()
    lift = height-rest[..., None]
    risk = (current_height-rest >= .03) & current_pair.bool()
    lost = torch.zeros_like(current_height, dtype=torch.long)
    loss = torch.zeros_like(pair)
    for j in range(32):
        lost = torch.where(pair[..., j], 0, lost+1)
        loss[..., j] = lost >= 6
    failure = ((lift < .02) | loss) & risk[..., None]
    values = []
    for end in (16, 32):
        region = slice(8, end)
        values.extend((pair[..., region].float().mean(-1),
                       ((lift[..., region] >= .03) & pair[..., region]).float().mean(-1),
                       failure[..., region].any(-1).float()))
    values.extend((((lift[..., 8:] < .02).any(-1) & risk).float(),
                   (lift[..., 8:] >= .03).float().mean(-1)))
    return torch.stack(values, -1), risk


def rolling_y(height, pair, rest, before_height, before_pair, offsets=OFFSETS):
    if height.shape != pair.shape or rest.shape != height.shape[:-1]:
        raise ValueError('trajectory identity mismatch')
    if not offsets or min(offsets) < 0 or max(offsets)+32 > height.shape[-1]:
        raise ValueError('unsupported future horizon; no padding or truncation')
    values, risks = [], []
    for offset in offsets:
        current_h = before_height if offset == 0 else height[..., offset-1]
        current_p = before_pair if offset == 0 else pair[..., offset-1]
        y, risk = short_y(current_h, current_p, height[..., offset:offset+32],
                          pair[..., offset:offset+32], rest)
        values.append(y); risks.append(risk)
    return torch.stack(values, -2), torch.stack(risks, -1)


def failure_events(height, pair, rest, qualification):
    """Separate first raw threshold event and failure AFTER Z qualification."""
    height = np.asarray(height); pair = np.asarray(pair, dtype=bool)
    lost = 0; first_raw = None; post = None
    for j, (h, contact) in enumerate(zip(height, pair), 1):
        lost = 0 if contact else lost+1
        failed = h-rest < .02 or lost >= 6
        if failed and first_raw is None: first_raw = j
        if failed and qualification >= 0 and j > qualification and post is None: post = j
    if qualification >= 0:
        kind = 'qualified_then_drop' if post is not None else 'qualified_no_later_drop'
    else:
        kind = 'unqualified_with_threshold_event' if first_raw is not None else 'unqualified_no_observed_threshold_event'
    return dict(first_raw_failure_step=first_raw, first_postqualification_drop_step=post,
                qualification_step=int(qualification), event_kind=kind)


def pair_signal(gap, joint_risk, offsets, event_step, epsilon=EPSILON):
    """Compare GOOD minus BAD factual paths; no common later decision state."""
    gap = np.asarray(gap); offsets = np.asarray(offsets); joint_risk = np.asarray(joint_risk, dtype=bool)
    if gap.shape != offsets.shape or joint_risk.shape != offsets.shape: raise ValueError('query shape')
    observed = joint_risk & (offsets < event_step) if event_step is not None else np.zeros_like(joint_risk)
    separated = observed & (np.abs(gap) > epsilon)
    correct = observed & (gap > epsilon)
    inverse = observed & (gap < -epsilon)
    def first(mask):
        ids = np.flatnonzero(mask)
        if not len(ids): return None
        j = int(ids[0])
        return dict(offset=int(offsets[j]), lead_steps=int(event_step-offsets[j]))
    qualifying = []
    for j in np.flatnonzero(correct & (offsets > 0)):
        if event_step-offsets[j] >= 8 and not inverse[j+1:].any(): qualifying.append(int(j))
    signal = first(np.isin(np.arange(len(gap)), qualifying))
    return dict(first_absolute_separation=first(separated), first_correct_separation=first(correct),
                inverse_offsets=offsets[inverse].tolist(), eligible_query_offsets=offsets[observed].tolist(),
                delayed_signal=signal,
                subsequent_eligible_queries_after_signal=int((observed & (offsets > signal['offset'])).sum()) if signal else None)
