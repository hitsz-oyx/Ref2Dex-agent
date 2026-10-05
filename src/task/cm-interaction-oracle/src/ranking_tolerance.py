"""Same-current-state ranking contracts; no trajectory stitching or Z surrogate.

Arrays are [states,candidates]. Missing counterfactuals must have mask=False.
GT utility ties are excluded from accuracy and counted explicitly. Prediction
 ties on GT-strict pairs earn 1/2 credit. Top1 uses first-index tie breaking.
"""
import numpy as np


def _panel(truth, prediction, mask):
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if truth.ndim != 2 or prediction.shape != truth.shape or mask.shape != truth.shape:
        raise ValueError('truth/prediction/mask must match [same_states,candidates]')
    if not np.isfinite(truth[mask]).all() or not np.isfinite(prediction[mask]).all():
        raise ValueError('observed candidate scores must be finite')
    return truth, prediction, mask


def same_state_ranking(truth, prediction, mask, *, tie_atol=0.):
    if not np.isfinite(tie_atol) or tie_atol < 0:
        raise ValueError('tie_atol must be finite and nonnegative')
    truth, prediction, mask = _panel(truth, prediction, mask)
    credits, strict_count, tied_count = [], 0, 0
    selected = np.full(len(truth), -1, dtype=int)
    regrets = np.full(len(truth), np.nan)
    states_with_two = 0
    for state, (gt, pred, valid) in enumerate(zip(truth, prediction, mask)):
        ids = np.flatnonzero(valid)
        if not len(ids):
            continue
        chosen = ids[np.argmax(pred[ids])]
        selected[state] = chosen
        # One observed candidate does not provide a ranking comparison.
        if len(ids) < 2:
            continue
        states_with_two += 1
        regrets[state] = float(gt[ids].max() - gt[chosen])
        for index, i in enumerate(ids):
            for j in ids[index+1:]:
                target, proposed = gt[i]-gt[j], pred[i]-pred[j]
                if abs(target) <= tie_atol:
                    tied_count += 1
                    continue
                strict_count += 1
                credits.append(.5 if abs(proposed) <= tie_atol else float(target*proposed > 0))
    return dict(pairwise_accuracy=float(np.mean(credits)) if credits else None,
                strict_pairs=strict_count, gt_tied_pairs=tied_count,
                states_with_comparisons=states_with_two,
                mean_top1_regret=float(np.nanmean(regrets)) if states_with_two else None,
                selected=selected.tolist(),
                top1_regret=[None if np.isnan(v) else float(v) for v in regrets],
                missing_candidates=int((~mask).sum()))


def within_state_shuffle(scores, mask, *, seed):
    """Placebo permutes candidate identity only among observed same-state rows."""
    scores = np.asarray(scores, dtype=float)
    scores, _, mask = _panel(scores, scores, mask)
    result = scores.copy()
    rng = np.random.default_rng(seed)
    for row in range(len(scores)):
        ids = np.flatnonzero(mask[row])
        result[row, ids] = scores[row, rng.permutation(ids)]
    return result


def inject_y_noise(y, mask, *, sigma, seed):
    """Preparation only: channel-unit iid Gaussian noise, no clipping/tuning.

    Evaluating noisy predictions offline measures ranking degradation only.
    Gain retention requires a fresh rolling intervention with real closed-loop Z.
    """
    y, mask = np.asarray(y, dtype=float), np.asarray(mask, dtype=bool)
    sigma = np.asarray(sigma, dtype=float)
    if y.ndim != 3 or mask.shape != y.shape[:2]:
        raise ValueError('Y must be [states,candidates,channels] with candidate mask')
    if sigma.ndim > 1 or (sigma.ndim == 1 and len(sigma) != y.shape[-1]):
        raise ValueError('sigma must be scalar or one value per channel')
    if not np.isfinite(sigma).all() or (sigma < 0).any() or not np.isfinite(y[mask]).all():
        raise ValueError('finite nonnegative sigma and observed Y required')
    result = y.copy()
    result[mask] += np.random.default_rng(seed).normal(size=result[mask].shape)*sigma
    return result


def assert_environment_oof(train_env, test_env):
    """All states/candidates from an environment belong to a single fold.

    Cross-seed callers must supply stable (seed,env) keys, as tuples, lists,
    or rows of an array. Scalar env ids only identify groups within one seed.
    Preserve component types instead of coercing mixed keys to NumPy strings.
    """
    def groups(values):
        result = set()
        for value in values:
            key = tuple(value) if isinstance(value, (tuple, list, np.ndarray)) else value
            if isinstance(key, tuple) and not key:
                raise ValueError('environment identity cannot be empty')
            try:
                result.add(key)
            except TypeError as error:
                raise ValueError('environment identity must be scalar or flat composite key') from error
        return result
    train, test = groups(train_env), groups(test_env)
    if not train or not test:
        raise ValueError('OOF requires nonempty train/test environment groups')
    overlap = train & test
    if overlap:
        raise ValueError('environment leakage between folds')
    return dict(train_environments=len(train), test_environments=len(test))


def closed_loop_gain_retention(base_z, oracle_z, learned_z, *, evaluation_ids):
    """Paired real rolling-arm outcomes; callers attest ids/order and protocol.

    No candidate-panel lookup is accepted as a substitute for actual rolling Z.
    Nonpositive oracle gain leaves retention undefined. This is descriptive,
    not uncertainty estimation or a gate decision.
    """
    arrays = [np.asarray(v, dtype=float) for v in (base_z, oracle_z, learned_z)]
    if any(v.ndim != 1 or v.shape != arrays[0].shape for v in arrays) or not len(arrays[0]):
        raise ValueError('real closed-loop arm Z arrays must be paired and nonempty')
    ids = list(evaluation_ids)
    if len(ids) != len(arrays[0]) or len(set(ids)) != len(ids):
        raise ValueError('unique paired evaluation ids required')
    if any(not np.isfinite(v).all() or not np.isin(v, [0., 1.]).all() for v in arrays):
        raise ValueError('closed-loop Z must be binary')
    base, oracle, learned = (float(v.mean()) for v in arrays)
    gain = oracle-base
    return dict(base_success=base, oracle_success=oracle, learned_success=learned,
                oracle_gain=gain, retention=(learned-base)/gain if gain > 0 else None,
                samples=len(ids), evidence_scope='real paired rolling outcomes, descriptive')
