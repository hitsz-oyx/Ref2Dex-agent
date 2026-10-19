import numpy as np
from ptv3_shared_wam_v5.event_labels import annotate_events, proximity_counts


def labels(values, valid=None):
    c = np.broadcast_to(np.asarray(values, np.float32)[:, None, None], (len(values), 1, 4))
    v = np.ones(c.shape[:2], bool) if valid is None else np.asarray(valid, bool)[:, None]
    enter, leave = proximity_counts(c, v)
    return annotate_events(enter, leave, v)


def test_initial_state_is_not_contact_and_short_noise_is_not_transition():
    a = labels([1.]*30 + [0., 0.] + [1.]*30)
    assert len(a['transitions']) == 0
    assert a['event_valid'][0, 0] and a['event_type'][0, 0] == 0
    assert not a['event_valid'][-1, 0]


def test_hysteresis_backdating_far_event_and_zero_delta():
    a = labels([0.]*40 + [1.]*5 + [.65]*5 + [0.]*30)
    assert a['transitions'].tolist() == [[40, 0, 1], [50, 0, 2]]
    assert a['event_type'][0, 0] == 1 and a['event_delta_steps'][0, 0] == 40
    assert a['event_type'][40, 0] == 1 and a['event_delta_steps'][40, 0] == 0
    assert a['event_type'][41, 0] == 2 and a['event_delta_steps'][41, 0] == 9
    assert np.all(a['state'][45:50] == 1)


def test_invalid_gap_does_not_create_release_and_censors_none():
    values = [1.]*35 + [0.] + [0.]*35
    valid = [True]*35 + [False] + [True]*35
    a = labels(values, valid)
    assert len(a['transitions']) == 0
    assert a['event_valid'][0, 0] and not a['event_valid'][15, 0]
    assert not a['event_valid'][35, 0] and a['event_valid'][36, 0]


def test_truncated_pending_onset_never_becomes_known_none():
    a = labels([0.]*26 + [1., 1.])
    assert not len(a['transitions'])
    assert a['event_valid'][0, 0]  # K24 ends before the pending onset.
    assert not a['event_valid'][2, 0]


def test_ambiguous_initial_band_and_multiple_slots():
    enter = np.zeros((40, 2), np.uint16)
    leave = np.ones((40, 2), np.uint16)*4
    enter[10:, 0] = 4
    leave[:, 1] = 0
    a = annotate_events(enter, leave, np.ones((40, 2), bool))
    assert np.all(a['state'][:10, 0] == -1)
    assert a['event_valid'][0, 1] and not a['event_valid'][0, 0]
    assert not len(a['transitions'])


def test_hoi4d_uses_observed_even_frames_but_preserves_30hz_timing():
    enter = np.zeros((100, 1), np.uint16); enter[40::2] = 4
    valid = np.zeros((100, 1), bool); valid[::2] = True
    a = annotate_events(enter, enter, valid, observation_stride=2)
    assert a['transitions'].tolist() == [[40, 0, 1]]
    assert a['event_delta_steps'][0, 0] == 40
    assert not a['event_valid'][1::2].any()
    valid[20, 0] = False
    b = annotate_events(enter, enter, valid, observation_stride=2)
    assert b['event_type'][0, 0] == 0  # no event looked up across missing observation
    assert not b['event_valid'][0, 0]  # censored before K24
