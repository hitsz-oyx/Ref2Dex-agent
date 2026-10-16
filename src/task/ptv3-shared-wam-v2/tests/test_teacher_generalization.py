import numpy as np
import pytest

from ptv3_shared_wam_v2.teacher_generalization import (
    assert_support_disjoint, isolated_middle, summarize,
)


def test_real_frame_support_is_disjoint_on_both_flanks():
    ticks = np.arange(200)
    train, held, interval = isolated_middle(ticks)
    assert interval == [80, 120]
    assert assert_support_disjoint(ticks[train], ticks[held]) == 28
    training_support = {f for t in ticks[train] for f in range(t - 3, t + 25)}
    held_support = {f for t in ticks[held] for f in range(t - 3, t + 25)}
    assert not training_support & held_support
    assert np.any(ticks[train] < 80) and np.any(ticks[train] >= 120)
    assert not np.any(train & held)


def test_noncontiguous_admission_uses_ticks_not_row_offsets():
    ticks = np.r_[np.arange(20), np.arange(100, 280), np.arange(400, 405)]
    train, held, interval = isolated_middle(ticks)
    assert interval == [172, 208]
    assert_support_disjoint(ticks[train], ticks[held])
    assert not np.any(np.isin(ticks[held], np.arange(20)))


def test_short_sequence_is_explicitly_training_only():
    train, held, interval = isolated_middle(np.arange(49))
    assert train.all() and not held.any() and interval is None


def test_audit_catches_distinct_queries_with_overlapping_windows():
    with pytest.raises(ValueError, match='overlaps'):
        assert_support_disjoint([53], [80])
    assert assert_support_disjoint([52], [80]) == 28


def test_sequence_weighted_metric_does_not_favor_long_sequences():
    result = summarize([1, 1, 1, 9], [0, 0, 0, 1])
    assert result['mean_mm'] == 3
    assert result['sequence_mean_mm'] == 5
