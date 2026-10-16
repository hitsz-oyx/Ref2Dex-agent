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


def phase_sampler(ids, sequence, tick, seed=181, batch_size=512):
    """Same sequence-uniform, half-startup draw as the original experiment."""
    ids = np.asarray(ids)
    if batch_size <= 0 or batch_size % 2:
        raise ValueError('Even positive batch required')
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
            seq = rng.integers(len(lengths), size=batch_size // 2)
            local = (rng.random(batch_size // 2) * lengths[seq]).astype(np.int64)
            batches.append(flat[offsets[seq] + local])
        return np.concatenate(batches)
    return draw


def balanced_sampler(original, expanded, sequence, tick):
    """Half original family, half other TRAIN; half startup within each half."""
    other = np.setdiff1d(expanded, original)
    first = phase_sampler(original, sequence, tick, batch_size=256)
    second = phase_sampler(other, sequence, tick, batch_size=256)
    def draw():
        x, y = first(), second()
        return np.r_[x[:128], y[:128], x[128:], y[128:]]
    return draw


def diverse_subset(ids, sequence, count, seed=181):
    """Equal label count, spread across TRAIN sequences without reading labels."""
    ids = np.asarray(ids)
    if not 0 < count <= len(ids):
        raise ValueError('Valid label budget required')
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(ids[sequence[ids] == s]) for s in np.unique(sequence[ids])]
    chosen = []
    for depth in range(max(map(len, pools))):
        for index in rng.permutation(len(pools)):
            if len(pools[index]) > depth:
                chosen.append(pools[index][depth])
                if len(chosen) == count:
                    return np.sort(chosen)
    raise AssertionError('Insufficient candidates')
