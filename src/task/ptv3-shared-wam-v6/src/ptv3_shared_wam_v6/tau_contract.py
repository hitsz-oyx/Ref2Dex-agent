"""CPU contracts for a future displacement channel, independently of a model."""
import numpy as np


def future_times(hz, steps=8):
    if not np.isscalar(hz) or not np.isfinite(hz) or hz <= 0 or steps < 1:
        raise ValueError('positive clock and future horizon required')
    return np.arange(1, steps + 1) / float(hz)


def gt_reference_tau(hand, tick, *, history=4, steps=8, encoder_available=True):
    """Reference index advances with tick; live hand/goal/event are not inputs.

    tau[0] = ref[t+1] - ref[t]. There is no current-time row in this array.
    H4 startup and missing future use the same-executor zero/false fallback.
    """
    hand = np.asarray(hand)
    if hand.ndim != 3 or hand.shape[1:] != (11, 3) or not np.isfinite(hand).all():
        raise ValueError('finite reference hand[N,11,3] required')
    if not isinstance(tick, (int, np.integer)) or tick < 0:
        raise ValueError('nonnegative integer reference tick required')
    valid = bool(encoder_available and history - 1 <= tick and tick + steps < len(hand))
    tau = hand[tick + np.arange(1, steps + 1)] - hand[tick] if valid else np.zeros((steps, 11, 3))
    return tau.astype(np.float32), valid


def reference_coverage(states, endpoint_tick, *, hz=30, steps=8, history=4):
    if states < 1 or not 0 <= endpoint_tick < states:
        raise ValueError('recorded endpoint required')
    valid = max(0, states - steps - (history - 1))
    last_valid = states - steps - 1 if valid else None
    return dict(states=states, controls=states-1, valid_tau_controls=valid,
                valid_control_fraction=valid/max(1, states-1),
                first_valid_tick=history-1 if valid else None, last_valid_tick=last_valid,
                endpoint_tick=endpoint_tick, post_endpoint_states=states-1-endpoint_tick,
                post_endpoint_elapsed_s=(states-1-endpoint_tick)/hz,
                valid_post_endpoint_elapsed_s=max(0, (last_valid or 0)-endpoint_tick)/hz,
                endpoint_plus_1s_tau_covered=bool(last_valid is not None and
                    last_valid >= endpoint_tick + int(np.ceil(hz))))


def first_loss_evidence(held, tau_valid, reference_states, *, steps=8, history=4):
    """First false held state after any held state; inspect the actual mask.

    This is loss of the narrow predicate, not necessarily FeedbackObserver's
    grasp_lost flag. Masks include post-action states; a terminal state with no
    following command must not be counted as a valid control.
    """
    held = np.asarray(held, bool)
    valid = np.asarray(tau_valid)
    if held.ndim != 1 or valid.ndim != 1 or len(valid) != len(held) or not np.isin(valid, [0, 1]).all():
        raise ValueError('aligned held states and binary actual tau mask required')
    seen = np.maximum.accumulate(held) if len(held) else held
    lost = np.flatnonzero(seen & ~held)
    tick = int(lost[0]) if len(lost) else None
    in_reference = bool(tick is not None and history-1 <= tick and tick+steps < reference_states)
    actual = bool(valid[tick]) if tick is not None else None
    return dict(first_loss_tick=tick, last_held_tick_before_first_loss=None if tick is None else tick-1,
                actual_tau_valid=actual,
                reference_future_available=in_reference,
                loss_while_gt_valid=bool(in_reference and actual),
                coverage_expiry_explains_first_loss=False if in_reference and actual else None)
