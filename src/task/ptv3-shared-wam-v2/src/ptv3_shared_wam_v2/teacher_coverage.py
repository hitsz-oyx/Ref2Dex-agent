"""Expand TRAIN coverage without changing the fixed temporal holdout."""
import numpy as np

from .teacher_generalization import assert_support_disjoint


def coverage_indices(sequence, tick, original_seen, held, records):
    sequence, tick = np.asarray(sequence), np.asarray(tick)
    original_seen, held = np.asarray(original_seen), np.asarray(held)
    allowed = np.ones(len(sequence), dtype=bool)
    for seq in np.unique(sequence[held]):
        rows = np.flatnonzero(sequence == seq)
        held_ticks = np.sort(tick[held[sequence[held] == seq]])
        position = np.searchsorted(held_ticks, tick[rows])
        distance = np.minimum(
            abs(tick[rows] - held_ticks[np.maximum(position - 1, 0)]),
            abs(tick[rows] - held_ticks[np.minimum(position, len(held_ticks) - 1)]))
        allowed[rows[distance <= 27]] = False
        assert_support_disjoint(tick[rows[allowed[rows]]], held_ticks)
    expanded = np.flatnonzero(allowed)
    cups = np.array([r['object'] in ('cup', 'mug') for r in records])
    if not np.array_equal(expanded[cups[sequence[expanded]]], original_seen):
        raise ValueError('Original cup/mug training membership changed')
    if np.intersect1d(expanded, held).size:
        raise ValueError('Holdout entered expanded training')
    return expanded


def phase_sampler(ids, sequence, tick, seed=181):
    """Same sequence-uniform, half-startup draw as the original experiment."""
    ids = np.asarray(ids)
    rng = np.random.default_rng(seed)
    phases = []
    for phase in (tick < 16, tick >= 16):
        pools = [ids[phase[ids] & (sequence[ids] == s)] for s in np.unique(sequence[ids])]
        pools = [p for p in pools if len(p)]
        if not pools:
            raise ValueError('Both phases required')
        lengths = np.array([len(p) for p in pools])
        phases.append((np.concatenate(pools), np.r_[0, np.cumsum(lengths)[:-1]], lengths))
    def draw():
        batches = []
        for flat, offsets, lengths in phases:
            seq = rng.integers(len(lengths), size=256)
            local = (rng.random(256) * lengths[seq]).astype(np.int64)
            batches.append(flat[offsets[seq] + local])
        return np.concatenate(batches)
    return draw
