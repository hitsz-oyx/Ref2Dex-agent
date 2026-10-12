"""The actual observation scheduler cannot receive a future RGB frame."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/audit'))
from rolling_video_depth import rolling_prefix_ids


def test_current_prefix_is_independent_of_future_label_schedule():
    current = rolling_prefix_ids(np.array([15]))[0]
    labels = rolling_prefix_ids(np.arange(15, 40, 3))
    np.testing.assert_array_equal(current, [12, 13, 14, 15])
    np.testing.assert_array_equal(labels[0], current)
    assert (labels <= np.arange(15, 40, 3)[:, None]).all()
    assert not np.array_equal(labels[1], current)


def test_missing_history_and_invalid_clock_rejected():
    for frames in (np.array([2, 5]), np.array([15, 15]), np.array([15., 18.])):
        with pytest.raises(ValueError):
            rolling_prefix_ids(frames)
