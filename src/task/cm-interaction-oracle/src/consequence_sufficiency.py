"""Frozen GT consequence/prognosis contract, not identified causal mediation."""
import numpy as np
import torch
from intervention import physical_targets, continuation_outcomes

HEADS = ('contact16', 'held16', 'combined_failure16', 'contact32', 'held32',
         'combined_failure32', 'height_failure32_late', 'height_held_fraction32_late')
PRIMARY = (3, 6, 7)
MODEL_SLOTS = {
    'H': (), 'Ha': ('a',), 'HE': ('e',), 'HI': ('i',),
    'HEI': ('e', 'i'), 'HaEI': ('a', 'e', 'i'),
    'HI_signed': ('i', 'signed'), 'HEI_signed': ('e', 'i', 'signed'),
    'HaEI_signed': ('a', 'e', 'i', 'signed'),
}


def readouts(p):
    """Mediators use only before and step8; all Y windows start at step9."""
    physical = physical_targets(p['before'], p['trajectory'][:, 7])
    outcome, details = continuation_outcomes(p['before'], p['trajectory'], p['rest_height'])
    lift = p['trajectory'][:, 8:, 2] - p['rest_height'][:, None]
    y = torch.cat((outcome, details['late_height_failure'][:, None].float(),
                   (lift >= .03).float().mean(1, keepdim=True)), -1)
    return physical[:, :12], physical[:, 12:], torch.asinh(p['trajectory'][:, 7, 48:66]), y, details


def matched_inputs(h, action, effect, interaction, signed):
    """Identical dimensions/init/capacity; absent features occupy zero slots."""
    values = dict(a=action, e=effect, i=interaction, signed=signed)
    return {name: torch.cat((h, *(v if key in active else torch.zeros_like(v)
                                for key, v in values.items())), -1)
            for name, active in MODEL_SLOTS.items()}


def supported_ranking(prediction, target, ids, groups, arms):
    """Factual prognosis only; exclude strata with <10 eligible cross-arm pairs."""
    result = {}
    for axis, tolerance in ((3, .03), (6, .5), (7, .03)):
        scores, support = [], []
        for group in np.unique(groups[ids]):
            rows = ids[groups[ids] == group]
            i, j = np.triu_indices(len(rows), 1)
            left, right = rows[i], rows[j]
            difference = target[left, axis] - target[right, axis]
            keep = (arms[left] != arms[right]) & (np.abs(difference) > tolerance)
            left, right, difference = left[keep], right[keep], difference[keep]
            support.append(dict(stratum=int(group), pairs=len(left), included=len(left) >= 10))
            if len(left) >= 10:
                product = (prediction[left, axis] - prediction[right, axis]) * difference
                scores.append(float(((product > 0) + .5 * (product == 0)).mean()))
        result[HEADS[axis]] = dict(accuracy=float(np.mean(scores)) if scores else None, support=support)
    return result


def cluster_gain(error_base, error_new, clusters, seed, repeats=2000):
    """Paired environment bootstrap on fixed held-out predictions; no model refit."""
    unique = np.unique(clusters)
    totals = np.array([[error_base[clusters == c].sum(), error_new[clusters == c].sum()]
                       for c in unique])
    rng = np.random.default_rng(seed)
    samples = totals[rng.integers(len(unique), size=(repeats, len(unique)))].sum(1)
    gains = 1 - samples[:, 1] / np.maximum(samples[:, 0], 1e-12)
    return dict(gain=float(1 - error_new.mean() / max(error_base.mean(), 1e-12)),
                lower95=float(np.quantile(gains, .025)), upper95=float(np.quantile(gains, .975)),
                upper_one_sided95=float(np.quantile(gains, .95)), clusters=len(unique), repeats=repeats)


def cross_half_bridge(coefficients):
    """Held-arm AND held-wave-half transfer; PCA3/ridge1 frozen without Y tuning.

    Contrasts share a zero reference and have noisy estimates. This tests a
    coarse consequence map, not per-state counterfactual sufficiency/mediation.
    """
    representations = dict(E=slice(0, 12), I=slice(12, 26), EI=slice(0, 26), EI_signed=slice(0, 44))
    output = {}
    for name, axes in representations.items():
        prediction, target, mean_base = [], [], []
        for source, destination in ((0, 1), (1, 0)):
            a, b = coefficients[source], coefficients[destination]
            for arm in range(14):
                keep = np.arange(14) != arm
                x, y = a[keep, axes], a[keep, 44:]
                scale = np.maximum(x.std(0), .001)
                x = x / scale
                _, _, v = np.linalg.svd(x, full_matrices=False)
                projection = v[:3].T
                z = x @ projection
                weight = np.linalg.solve(z.T @ z + np.eye(3), z.T @ y)
                prediction.append(b[arm, axes] / scale @ projection @ weight)
                target.append(b[arm, 44:])
                mean_base.append(y.mean(0))
        prediction, target, mean_base = map(np.asarray, (prediction, target, mean_base))
        mse = ((prediction - target) ** 2).mean(0)
        zero_mse, mean_mse = (target ** 2).mean(0), ((mean_base - target) ** 2).mean(0)
        correlations = [float(np.corrcoef(prediction[:, j], target[:, j])[0, 1])
                        if min(prediction[:, j].std(), target[:, j].std()) > 1e-12 else None
                        for j in range(8)]
        output[name] = dict(predictions=prediction.tolist(), target=target.tolist(),
            gain_vs_zero=(1 - mse / np.maximum(zero_mse, 1e-12)).tolist(),
            gain_vs_source_mean=(1 - mse / np.maximum(mean_mse, 1e-12)).tolist(),
            correlation=correlations, folds=28, components=3, ridge=1.,
            limits='14 nonzero arm estimates, shared zero, two halves; descriptive noisy aggregate transfer')
    return output
