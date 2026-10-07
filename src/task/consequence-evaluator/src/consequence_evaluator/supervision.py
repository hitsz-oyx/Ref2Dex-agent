"""Auditable local physical-event preferences, never episode-outcome labels.

Contact is the native net-force proxy, not a measured hand/object contact pair.
The thresholds below adapt the repository's sustained-lift task criterion.
They do not reproduce Robometer's video labels or DenseReward's reward curves.
"""
import hashlib

import numpy as np

from .contracts import K, K_EXEC, HISTORY_MATCH_MAX_RELATIVE_RMS

RULE = 'geometry-corroborated-H-matched-state-residual-events-v3'
CONTACT_SEMANTICS = 'native_hand_and_object_net_force_proxy'


def consecutive(mask):
    mask = np.asarray(mask, dtype=bool)
    result = np.zeros(len(mask), dtype=np.int64)
    for index, value in enumerate(mask):
        result[index] = (result[index - 1] if index else 0) + 1 if value else 0
    return result


def physical_trace(packet, diagnostics, *, require_geometry=True):
    """Validate post-action diagnostics before they may generate supervision."""
    steps = len(packet['action'])
    pose = packet['object_pose']
    contact, valid = diagnostics['contact'], diagnostics['contact_valid']
    if (pose.shape != (steps + 1, 4, 4) or not np.isfinite(pose).all()
            or contact.shape != (steps + 1,) or valid.shape != (steps + 1,)
            or contact.dtype != np.bool_ or valid.dtype != np.bool_ or not valid[1:].all()
            or valid[0] or not np.isclose(float(diagnostics['initial_height']), pose[0, 2, 3], atol=1e-5)):
        raise ValueError('invalid physical-label diagnostics or initial contact validity')
    height = pose[:, 2, 3] - pose[0, 2, 3]
    gap=None
    corroborated=contact
    if require_geometry:
        gap=np.asarray(diagnostics.get('surface_gap'))
        if gap.shape!=(steps+1,) or not np.isfinite(gap).all() or (gap<0).any():
            raise ValueError('measured hand/object sampled-surface gap required for event labels')
        corroborated=contact & (gap<=.01)
    held = valid & corroborated & (height >= .03)
    lost = consecutive(valid & ~corroborated)
    return dict(height=height, contact=corroborated, valid=valid, held=held,
                force_proxy=contact,surface_gap=gap,
                held_run=consecutive(held), drop=(height < .02) | (lost >= 6))


def expert_anchor(trace, record):
    """A clean, verified 45-frame hold with no subsequent drop anchors progress."""
    stable = np.flatnonzero(trace['held_run'] >= 45)
    completion = int(stable[0]) if len(stable) else None
    clean = record['assigned_phase'] == 'clean' and record['perturbation_tick'] == -1
    reliable = completion is not None and not trace['drop'][completion + 1:].any()
    quality = ('expert_success' if clean else 'perturbed_success') if reliable else 'suboptimal' if len(stable) else 'failure'
    progress = np.full(len(trace['height']), np.nan, dtype='float32')
    mask = np.zeros(len(progress), dtype=bool)
    if quality == 'expert_success':
        # Absolute episode scale through verified completion, then saturation.
        # Never renormalize a 24-step slice to 0..1.
        progress = np.minimum(np.arange(len(progress)) / completion, 1).astype('float32')
        mask[:] = True
    return quality, progress, mask, completion


def local_event(trace, phase, tick):
    """Abstain from ambiguous miss/recovery events; read only t..t+24."""
    stop = tick + K + 1
    if tick < 0 or stop > len(trace['height']) or not trace['valid'][tick:stop].all():
        return None
    held = trace['held'][tick:stop]
    contact = trace['contact'][tick:stop]
    height = trace['height'][tick:stop]
    if phase in ('lift', 'hold') and held[0]:
        if held.all():
            return 'maintained_hold'
        # Lost-contact evidence uses only this window, not a preceding streak.
        drop = (height[1:] < .02) | (consecutive(~contact[1:]) >= 6)
        if drop.any():
            first = int(np.flatnonzero(drop)[0]) + 1
            if not held[first:].any() and drop[-1]:
                return 'unrecovered_drop'
    if phase == 'grasp' and contact[0] and height[0] < .03:
        if held[-K_EXEC:].all():
            return 'lift_achieved'
        if not held[1:].any() and not contact[-K_EXEC:].any():
            return 'grasp_lost'
    return None


def order_key(window):
    identity = window['episode'] + ':' + str(window['tick'])
    return hashlib.sha256(identity.encode()).hexdigest()


def states_match(first,second, *, require_history=False):
    """Match current physical state and, when present, the full policy history H.

    Full object z additionally matches within1cm. Same motion/reference supplies
    the shared reset height; this is observational matching, never a fork. A
    relative history gate prevents velocity/contact-memory confounds from being
    silently used by the baseline arm.
    """
    a,b=np.asarray(first['object_pose']),np.asarray(second['object_pose'])
    pa,pb=np.asarray(first['hand_keypoints']),np.asarray(second['hand_keypoints'])
    if (a.shape!=(4,4) or b.shape!=(4,4) or pa.shape!=(11,3) or pb.shape!=(11,3)
            or not all(np.isfinite(v).all() for v in (a,b,pa,pb))):
        raise ValueError('current object pose and measured11-point hand geometry required')
    angle=np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1))
    physical = (np.linalg.norm(a[:3,3]-b[:3,3])<=.03 and abs(a[2,3]-b[2,3])<=.01
                and angle<=np.deg2rad(15) and np.sqrt(np.mean(np.sum((pa-pb)**2,axis=-1)))<=.02)
    if not physical:
        return False
    if require_history and (('history' not in first) or ('history' not in second)):
        return False
    if ('history' in first) != ('history' in second):
        return False
    if 'history' not in first:
        return True
    ha, hb = np.asarray(first['history']), np.asarray(second['history'])
    if ha.shape != hb.shape or not np.isfinite(ha).all() or not np.isfinite(hb).all():
        return False
    scale = np.maximum(1., np.maximum(np.abs(ha), np.abs(hb)))
    relative_rms = float(np.sqrt(np.mean(((ha - hb) / scale) ** 2)))
    return relative_rms <= HISTORY_MATCH_MAX_RELATIVE_RMS


def pair_coverage_ready(counts):
    """Check the minimum unique local preferences required for a Probe fit."""
    from .contracts import MIN_PREFERENCE_PAIRS
    return all(int(counts.get(split, 0)) >= minimum
               for split, minimum in MIN_PREFERENCE_PAIRS.items())


def local_preferences(windows, per_stratum=64):
    """Deterministic, capped, cross-episode phase-matched comparisons.

    Current relative height must match within 1cm. At most two comparisons per
    unordered episode pair prevent one long episode from filling a stratum.
    No quality or eventual-outcome field is consulted.
    """
    if not 1 <= per_stratum <= 64:
        raise ValueError('at most64 preferences per split/task/phase/event stratum')
    buckets = {}
    for window in windows:
        key = (window['split'], window['task'],window['expert'],window['motion'], window['phase'])
        buckets.setdefault(key, []).append(window)
    pairs = []
    for key, group in sorted(buckets.items()):
        for good_event, bad_event in (('maintained_hold', 'unrecovered_drop'), ('lift_achieved', 'grasp_lost')):
            good = sorted((w for w in group if w['event'] == good_event), key=order_key)
            bad = sorted((w for w in group if w['event'] == bad_event), key=order_key)
            if not bad:
                continue
            heights = np.asarray([w['initial_relative_height'] for w in bad])
            counts, count = {}, 0
            for chosen in good:
                distances = np.abs(heights - chosen['initial_relative_height'])
                for index in np.argsort(distances, kind='stable'):
                    rejected = bad[index]
                    identity = tuple(sorted((chosen['episode'], rejected['episode'])))
                    if distances[index] > .01:
                        break
                    if chosen['episode'] == rejected['episode'] or counts.get(identity, 0) >= 2:
                        continue
                    if not states_match(chosen,rejected, require_history=True):continue
                    pairs.append(dict(chosen={k: chosen[k] for k in ('episode', 'tick')},
                                      rejected={k: rejected[k] for k in ('episode', 'tick')},
                                      annotation=f'{RULE}: {good_event}>{bad_event}; actual t..t+24; '
                                                 'force proxy corroborated by<=1cm sampled surface gap; '
                                                 'same expert/motion; current object3cm/15deg and hand2cm RMS; '
                                                 'history relative RMS<=25% match'))
                    counts[identity] = counts.get(identity, 0) + 1
                    count += 1
                    break
                if count >= per_stratum:
                    break
    return pairs
