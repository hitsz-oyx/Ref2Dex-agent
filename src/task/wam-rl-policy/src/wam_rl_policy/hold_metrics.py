"""Realtime task-success/hold metrics, separate from the upstream ever flag."""
from __future__ import annotations

import numpy as np


def summarize_realtime_hold(distance, threshold=.05):
    """Summarize a per-tick goal-distance sequence.

    ``distance`` should include the final state after the last executed
    action when the caller can observe it.  The upstream ``successes`` tensor
    is a cumulative "ever reached" flag; it cannot measure a contiguous hold.
    """
    values = np.asarray(distance, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty goal-distance sequence required")
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("positive finite success threshold required")
    near = values <= threshold
    longest = current = 0
    for flag in near:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    first = int(np.argmax(near)) if near.any() else None
    drops = int(np.count_nonzero(near[:-1] & ~near[1:])) if len(near) > 1 else 0
    onsets = int(np.count_nonzero(~near[:-1] & near[1:])) if len(near) > 1 else 0
    return dict(
        success_frames=int(near.sum()),
        longest_hold_frames=int(longest),
        first_success_index=first,
        final_success=bool(near[-1]),
        ever_success=bool(near.any()),
        success_onsets=onsets,
        drop_events_after_success=drops,
        final_distance_m=float(values[-1]),
        min_distance_m=float(values.min()),
    )
