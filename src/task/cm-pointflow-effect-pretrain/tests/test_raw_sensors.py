"""No future-maximum leakage or invented calibrated hand frame."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oakink_wm.raw_sensors import sensor_window


def source():
    raw = np.tile(np.arange(512) % 256, (32, 1)).astype('float32')
    joints = np.random.RandomState(0).normal(size=(32, 2, 21, 3)).astype('float32')
    return raw, joints, np.arange(32) / 30.


def test_future_sensor_peak_cannot_change_history_or_hand_inputs():
    raw, joints, ts = source()
    a = sensor_window(raw, joints, ts, 0)
    raw[4:] = 255
    b = sensor_window(raw, joints, ts, 0)
    for key in ('sensor_history', 'sensor_current', 'hand_history', 'future_hand_shape'):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a['target_delta'], b['target_delta'])
    assert a['sensor_history'][0, 255] == 1
    assert abs(a['elapsed'][-1] - .8) < 1e-6


def test_future_hand_size_cannot_rescale_observed_shape():
    raw, joints, ts = source()
    a = sensor_window(raw, joints, ts, 0)
    joints[4:] *= 20
    b = sensor_window(raw, joints, ts, 0)
    np.testing.assert_array_equal(a['hand_history'], b['hand_history'])
    assert not np.array_equal(a['future_hand_shape'], b['future_hand_shape'])
    translated = joints + 100
    c = sensor_window(raw, translated, ts, 0)
    np.testing.assert_allclose(b['hand_history'], c['hand_history'], atol=2e-5)


def test_missing_hands_and_nonmonotonic_original_clock_are_rejected():
    raw, joints, ts = source()
    joints[2, 0] = np.nan
    with pytest.raises(ValueError, match='invalid hand'):
        sensor_window(raw, joints, ts, 0)
    _, joints, ts = source()
    ts[1] = ts[0]
    with pytest.raises(ValueError, match='invalid hand/clock'):
        sensor_window(raw, joints, ts, 0)
