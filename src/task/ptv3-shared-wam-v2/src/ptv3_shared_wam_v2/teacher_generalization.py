"""Temporal membership for the full-teacher control generalization probe."""
import numpy as np


def isolated_middle(ticks, history=3, future=24):
    """Hold out the middle fifth of the longest contiguous admitted run.

    Keep training queries only when their closed [t-history,t+future]
    support cannot overlap any holdout query's support. No labels are read.
    Short sequences remain training-only and are explicitly identified.
    """
    ticks = np.asarray(ticks, dtype=np.int64)
    if len(ticks) == 0 or np.any(np.diff(ticks) <= 0):
        raise ValueError('nonempty strictly increasing ticks required')
    runs = np.split(ticks, np.flatnonzero(np.diff(ticks) != 1) + 1)
    run = max(runs, key=len)
    start = int(run[0] + len(run) * .4)
    stop = int(run[0] + len(run) * .6)
    hold = (ticks >= start) & (ticks < stop)
    train = (ticks < start - history - future) | (ticks > stop - 1 + history + future)
    # Both temporal flanks must exist: this is interpolation, not a prefix test.
    if (stop - start < 16 or np.sum(train & (ticks < start)) < 8
            or np.sum(train & (ticks >= stop)) < 8):
        return np.ones(len(ticks), dtype=bool), np.zeros(len(ticks), dtype=bool), None
    return train, hold, [start, stop]


def assert_support_disjoint(train_ticks, held_ticks, history=3, future=24):
    """Audit actual membership independently of the splitting formula."""
    train_ticks, held_ticks = np.asarray(train_ticks), np.sort(held_ticks)
    if not len(train_ticks) or not len(held_ticks):
        raise ValueError('both training and interpolation samples required')
    position = np.searchsorted(held_ticks, train_ticks)
    nearest = np.minimum(
        abs(train_ticks - held_ticks[np.maximum(position - 1, 0)]),
        abs(train_ticks - held_ticks[np.minimum(position, len(held_ticks) - 1)]))
    if np.any(nearest <= history + future):
        raise ValueError('training/holdout frame support overlaps')
    return int(nearest.min())


def summarize(errors, sequences):
    """Report both tick-weighted and sequence-weighted physical errors."""
    errors, sequences = np.asarray(errors), np.asarray(sequences)
    if not len(errors):
        return dict(count=0, sequences=0, mean_mm=None, p95_mm=None,
                    sequence_mean_mm=None)
    means = [float(errors[sequences == s].mean()) for s in np.unique(sequences)]
    return dict(count=len(errors), sequences=len(means), mean_mm=float(errors.mean()),
                median_mm=float(np.median(errors)), p95_mm=float(np.percentile(errors, 95)),
                sequence_mean_mm=float(np.mean(means)))
