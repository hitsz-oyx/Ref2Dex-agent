"""Paired fixed-panel uncertainty with whole mother-sequence resampling."""
import numpy as np


def cluster_median_ci(values, sequence_ids, *, cluster_universe, seed, repeats=2000):
    """Resample mothers, preserving all their windows/frames and multiplicity.

    The statistic remains the pooled frame median, NOT a median of sequence
    medians. Mothers with no frames in a subset remain in the panel universe.
    """
    values = np.asarray(values, dtype=np.float64)
    sequence_ids = np.asarray(sequence_ids)
    universe = np.unique(cluster_universe)
    if values.ndim != 1 or sequence_ids.shape != values.shape or not np.isfinite(values).all():
        raise ValueError('finite paired values with one mother id per value required')
    if repeats < 2000 or not len(universe):
        raise ValueError('at least 2000 mother-cluster bootstrap replicates required')
    if not len(values):
        return dict(median_mm=None, ci95_mm=[None, None], bootstrap_repeats=repeats,
                    cluster_count=len(universe), clusters_with_values=0)
    groups = np.searchsorted(universe, sequence_ids)
    if np.any(groups >= len(universe)) or not np.array_equal(universe[groups], sequence_ids):
        raise ValueError('value mother absent from cluster universe')
    order = np.argsort(values)
    ordered, groups = values[order], groups[order]
    rng = np.random.default_rng(seed)
    medians = []
    for _ in range(repeats):
        counts = np.bincount(rng.integers(len(universe), size=len(universe)), minlength=len(universe))
        cumulative = np.cumsum(counts[groups])
        n = int(cumulative[-1])
        if not n:
            continue
        lower = np.searchsorted(cumulative, (n-1)//2, side='right')
        upper = np.searchsorted(cumulative, n//2, side='right')
        medians.append((ordered[lower]+ordered[upper])/2)
    if len(medians) < repeats*.95:
        raise ValueError('too many empty-subset bootstrap replicates; report insufficient clusters')
    return dict(median_mm=float(np.median(values)),
                ci95_mm=np.percentile(medians, [2.5, 97.5]).tolist(),
                bootstrap_repeats=repeats, nonempty_repeats=len(medians), bootstrap_seed=int(seed),
                resampling_unit='mother sequence_id; all windows/frames retained',
                statistic='pooled frame median of paired B1 minus B0a errors',
                cluster_count=len(universe), clusters_with_values=len(np.unique(sequence_ids)))
