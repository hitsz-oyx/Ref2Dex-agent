import numpy as np
import pytest

from ptv3_shared_wam_v2.teacher_coverage import coverage_indices, phase_sampler, diverse_subset, balanced_sampler
from ptv3_shared_wam_v2.teacher_generalization import isolated_middle


def test_added_object_preserves_cup_support_and_original_rows():
    ticks = np.tile(np.arange(200), 2)
    sequence = np.repeat([0, 1], 200)
    keep, held, _ = isolated_middle(np.arange(200))
    seen = np.flatnonzero(keep)
    expanded = coverage_indices(sequence, ticks, seen, np.flatnonzero(held),
                                [{'object': 'cup'}, {'object': 'ball'}])
    assert np.array_equal(expanded[expanded < 200], seen)
    assert np.array_equal(expanded[expanded >= 200], np.arange(200, 400))
    with pytest.raises(ValueError, match='membership'):
        coverage_indices(sequence, ticks, seen[1:], np.flatnonzero(held),
                         [{'object': 'cup'}, {'object': 'ball'}])


def test_sampler_never_draws_holdout_and_retains_half_startup():
    ticks = np.arange(200)
    keep, held, _ = isolated_middle(ticks)
    sampler = phase_sampler(np.flatnonzero(keep), np.zeros(200), ticks)
    for _ in range(10):
        draw = sampler()
        assert len(draw) == 512
        assert (ticks[draw[:256]] < 16).all()
        assert (ticks[draw[256:]] >= 16).all()
        assert keep[draw].all()


def test_equal_label_budget_covers_sequences_without_new_rows():
    sequence = np.repeat([0, 1, 2], [2, 40, 60])
    legal = np.r_[np.arange(2), np.arange(10, 30), np.arange(50, 90)]
    rows = diverse_subset(legal, sequence, 20)
    assert len(rows) == len(np.unique(rows)) == 20
    assert np.isin(rows, legal).all()
    assert set(sequence[rows]) == {0, 1, 2}
    assert (sequence[rows] == 0).sum() == 2
    assert np.array_equal(rows, diverse_subset(legal, sequence, 20))


def test_balanced_sampler_preserves_family_and_phase_exposure():
    sequence = np.repeat(np.arange(3), 100)
    ticks = np.tile(np.arange(100), 3)
    sampler = balanced_sampler(np.arange(100), np.arange(300), sequence, ticks)
    for _ in range(10):
        rows = sampler()
        assert (rows < 100).sum() == 256
        assert (ticks[rows[:256]] < 16).all()
        assert (ticks[rows[256:]] >= 16).all()
