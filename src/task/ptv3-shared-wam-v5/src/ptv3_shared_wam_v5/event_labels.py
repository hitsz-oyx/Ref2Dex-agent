"""Offline proximity-event annotations. No spatial search or online lookahead."""
import numpy as np

EVENT_LABEL_SCHEMA = 'ref2dex.wam-v5-proximity-events.v2'
DEFAULT_PROTOCOL = dict(enter_confidence=.8, leave_confidence=.5, min_queries=3,
                        persistence=3, horizon=24, hz=30)
OBSERVATION_STRIDE_BY_SOURCE = {10: 2}  # HOI4D odd midpoints have no contact teacher.


def validate_protocol(protocol):
    if (set(protocol) != set(DEFAULT_PROTOCOL) or protocol['hz'] != 30
            or not 0 < protocol['leave_confidence'] < protocol['enter_confidence'] <= 1
            or protocol['min_queries'] < 1 or protocol['persistence'] < 1
            or protocol['horizon'] != 24):
        raise ValueError('explicit 30Hz K24 hysteresis/persistence protocol required')


def proximity_counts(confidence, valid, protocol=None, chunk_frames=128):
    protocol = DEFAULT_PROTOCOL if protocol is None else protocol
    validate_protocol(protocol)
    if confidence.shape[:2] != valid.shape or confidence.ndim != 3 or valid.ndim != 2:
        raise ValueError('frame/object/query contact and frame/object validity required')
    enter = np.zeros(valid.shape, np.uint16)
    leave = np.zeros_like(enter)
    for start in range(0, len(valid), chunk_frames):
        c = np.asarray(confidence[start:start+chunk_frames])
        v = valid[start:start+chunk_frames]
        if not np.isfinite(c[v]).all() or ((c[v] < 0) | (c[v] > 1)).any():
            raise ValueError('valid proximity confidences must be finite in [0,1]')
        enter[start:start+chunk_frames] = np.where(v, (c >= protocol['enter_confidence']).sum(-1), 0)
        leave[start:start+chunk_frames] = np.where(v, (c >= protocol['leave_confidence']).sum(-1), 0)
    return enter, leave


def annotate_events(enter_count, leave_count, valid, protocol=None, observation_stride=1):
    """Backdate sustained transitions; gaps and unobserved boundaries stay unknown.

    State -1/0/1 means unknown/clear/near. The initial sustained run establishes
    state without inventing an onset. Known NONE certifies only the following
    K24, with persistence-1 extra frames so pending onsets cannot be hidden.
    """
    p = dict(DEFAULT_PROTOCOL if protocol is None else protocol)
    validate_protocol(p)
    if observation_stride not in (1, 2):
        raise ValueError('only explicit native or HOI4D observed cadence is supported')
    valid = np.asarray(valid, bool)
    enter_count, leave_count = np.asarray(enter_count), np.asarray(leave_count)
    if valid.ndim != 2 or enter_count.shape != valid.shape or leave_count.shape != valid.shape:
        raise ValueError('aligned frame/object count and validity arrays required')
    if (enter_count > leave_count).any():
        raise ValueError('enter count cannot exceed lower-threshold count')
    if observation_stride == 2:
        if valid[1::2].any():
            raise ValueError('HOI4D cadence requires masked synthetic odd midpoints')
        coarse = _annotate(enter_count[::2], leave_count[::2], valid[::2], p, 12)
        result = {}
        for name in ('state', 'event_type', 'event_delta_steps', 'event_valid'):
            values = coarse[name]
            result[name] = np.full(valid.shape, -1 if name == 'state' else 0, values.dtype)
            result[name][::2] = values*2 if name == 'event_delta_steps' else values
        result['transitions'] = coarse['transitions'].copy()
        result['transitions'][:, 0] *= 2
        # Scheduler annotations can be queried between two real observations;
        # this does not synthesize a proximity measurement/state or C target.
        # Onsets remain quantized to even observed frames. Require both ends
        # of this one-observation interval known, and the right future packet
        # uncensored. Never carry across a missing observed frame.
        odd = np.arange(1, len(valid)-1, 2)
        bridge = ((coarse['state'][:-1] >= 0) & (coarse['state'][1:] >= 0)
                  & valid[odd-1] & valid[odd+1] & coarse['event_valid'][1:])
        for name in ('event_type', 'event_delta_steps', 'event_valid'):
            right = result[name][odd+1].copy()
            if name == 'event_delta_steps':
                right += (result['event_type'][odd+1] != 0)
            result[name][odd] = np.where(bridge, right, 0)
        return result
    return _annotate(enter_count, leave_count, valid, p, p['horizon'])


def _annotate(enter_count, leave_count, valid, p, observed_horizon):
    n, slots = valid.shape
    state = np.full((n, slots), -1, np.int8)
    events = []
    dwell, minimum = p['persistence'], p['min_queries']
    for slot in range(slots):
        current, pending, run = -1, -1, 0
        for t in range(n):
            if not valid[t, slot]:
                current, pending, run = -1, -1, 0
                continue
            proposal = (1 if enter_count[t, slot] >= minimum else
                        0 if leave_count[t, slot] < minimum else -1)
            state[t, slot] = current
            if proposal == -1 or proposal == current:
                pending, run = -1, 0
                continue
            pending, run = (proposal, run+1) if proposal == pending else (proposal, 1)
            if run == dwell:
                onset = t-dwell+1
                if current != -1:
                    events.append((onset, slot, 1 if proposal else 2))
                current = proposal
                state[onset:t+1, slot] = current
                pending, run = -1, 0
    kind = np.zeros(valid.shape, np.uint8)
    delta = np.zeros(valid.shape, np.int32)
    known = np.zeros(valid.shape, bool)
    by_slot = [dict() for _ in range(slots)]
    for t, slot, event in events:
        by_slot[slot][t] = event
    # Reverse scan resets at every unknown/gap. Far future events retain their
    # actual time; NONE never substitutes for a censored tail or invalid gap.
    for slot in range(slots):
        next_t, next_kind, end = None, 0, n
        for t in range(n-1, -1, -1):
            if state[t, slot] == -1 or not valid[t, slot]:
                next_t, next_kind, end = None, 0, t
                continue
            if t in by_slot[slot]:
                next_t, next_kind = t, by_slot[slot][t]
            if next_t is not None:
                kind[t, slot], delta[t, slot], known[t, slot] = next_kind, next_t-t, True
            elif t+observed_horizon+dwell-1 < end:
                known[t, slot] = True
    return dict(state=state, event_type=kind, event_delta_steps=delta, event_valid=known,
                transitions=np.asarray(sorted(events), np.int32).reshape(-1, 3))


def summarize_events(arrays, raw_valid):
    known, kind = arrays['event_valid'], arrays['event_type']
    return dict(frames=int(raw_valid.size), raw_valid=int(raw_valid.sum()),
                state_known=int((arrays['state'] >= 0).sum()), condition_valid=int(known.sum()),
                none=int((known & (kind == 0)).sum()),
                contact=int((known & (kind == 1)).sum()), release=int((known & (kind == 2)).sum()),
                far=int((known & (arrays['event_delta_steps'] > 24)).sum()),
                contact_transitions=int((arrays['transitions'][:, 2] == 1).sum()),
                release_transitions=int((arrays['transitions'][:, 2] == 2).sum()))
