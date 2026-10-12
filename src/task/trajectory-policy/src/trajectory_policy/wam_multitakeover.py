"""Admit the audited takeover dataset and balance development sampling."""
import numpy as np


def validate_aggregation(manifest, audit, data, warm_pair):
    valid = data['trajectory_valid']
    if (manifest.get('status') != 'COMPLETED' or manifest.get('training_allowed') is not True
            or audit.get('pass') is not True or audit.get('aggregation_sha256') != manifest.get('aggregation_sha256')
            or manifest.get('samples') != 643 or manifest.get('true_expert_future_samples') != 259
            or manifest.get('pure_current_PD_samples') != 384
            or valid.dtype != bool or valid.shape != (643,)
            or int(valid.sum()) != 259 or data['tau24x33'].shape != (643, 24, 33)
            or any(not np.isfinite(v).all() for v in data.values())
            or data['tau24x33'][~valid].any()
            or not np.array_equal(data['role'], np.where(valid, 3, 2))
            or manifest.get('model_identities') != [list(warm_pair)]*3):
        raise ValueError('audited exact-policy643 aggregation and pure-future exclusion required')
    admissions = manifest.get('admissions', [])
    if [v.get('takeover') for v in admissions] != [8, 16, 32]:
        raise ValueError('ordered actual takeover8/16/32 required')
    for sid, takeover in enumerate((8, 16, 32)):
        rows = data['source_id'] == sid
        future = rows & valid
        pure = rows & ~valid
        if (not np.array_equal(data['tick'][future], np.arange(takeover, 105))
                or not np.array_equal(data['tick'][pure], np.arange(128))
                or admissions[sid].get('pure_future_training_allowed') is not False
                or admissions[sid].get('actual_expert_future_training_allowed') is not True):
            raise ValueError('own-source future/current-PD scope differs')
    return dict(admitted=True, samples=643, actual_future=259, pure_current_PD=384,
                pure_future_training_allowed=False, R_supervision='valid actual expert future only')


def balanced_draw(rng, pools):
    """16 entries, rotating the extra slot across the three actual sources."""
    extra = int(rng.integers(3))
    return np.concatenate([rng.choice(pool, 6 if i == extra else 5)
                           for i, pool in enumerate(pools)])
