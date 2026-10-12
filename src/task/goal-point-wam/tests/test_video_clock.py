"""A source clip starting at zero can still contain valid later H4/K24 windows."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/audit'))
from video_clock import cached_window_clock


@pytest.mark.parametrize('offset,expected_prefix', [(3, [0, 1, 2, 3]), (6, [3, 4, 5, 6])])
def test_initial_clip_has_valid_observed_history_at_later_ticks(offset, expected_prefix):
    frames = np.arange(0, 31, 3)
    current, label_offset, prefix = cached_window_clock(frames, offset)
    np.testing.assert_array_equal(prefix, expected_prefix)
    assert current == frames[label_offset]
    assert current + 24 <= frames[-1]
    assert prefix.max() == current


def test_initial_tick_cannot_invent_missing_past():
    with pytest.raises(ValueError, match='three past'):
        cached_window_clock(np.arange(0, 31, 3), 0)


def test_existing_nonzero_clip_keeps_original_clock():
    current, label_offset, prefix = cached_window_clock(np.arange(15, 46, 3), 6)
    assert current == 21 and label_offset == 2
    np.testing.assert_array_equal(prefix, [18, 19, 20, 21])


@pytest.mark.parametrize('frames,offset', [
    (np.arange(0, 31, 3).astype(float), 3),
    (np.arange(0, 30, 3), 3),
    (np.arange(0, 33, 3)*2, 3),
    (np.arange(-3, 28, 3), 6),
    (np.arange(0, 31, 3), 9),
])
def test_invalid_measurement_clocks_and_uncovered_windows_rejected(frames, offset):
    with pytest.raises(ValueError):
        cached_window_clock(frames, offset)
