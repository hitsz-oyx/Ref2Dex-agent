"""Versioned native geometry and window cleaning; never modify source payloads."""
from dataclasses import asdict, dataclass

import numpy as np

from .windows import validate_sequence, xyzquat_to_matrix

CLEANING_SCHEMA = 'ref2dex.pose-mesh-clean.v1'


@dataclass(frozen=True)
class CleaningPolicy:
    stride: int = 8
    min_future_hand_fraction: float = .9
    alignment_limit_ms: float = 16.
    normal_min_norm: float = 1e-6
    drop_nonquery_zero_normals: bool = True

    def __post_init__(self):
        if not 1 <= self.stride <= 24:
            raise ValueError('stride must be in [1,24]')
        if not 0 < self.min_future_hand_fraction <= 1:
            raise ValueError('positive bounded hand coverage required')
        if not 0 < self.alignment_limit_ms <= 1000 / 60:
            raise ValueError('alignment must not exceed half a 30Hz period')
        if not np.isfinite(self.normal_min_norm) or self.normal_min_norm <= 0:
            raise ValueError('positive finite normal threshold required')
        if type(self.drop_nonquery_zero_normals) is not bool:
            raise ValueError('explicit bool normal filtering policy required')

    def to_dict(self):
        return asdict(self)


def normalize_geometry(values, policy):
    """Keep query identity; drop ambiguous nonquery points without guessing a sign."""
    result = {k: np.asarray(v).copy() for k, v in values.items()}
    surface, normals = result['surface'], result['normals']
    if (surface.ndim != 2 or surface.shape[1] != 3 or normals.shape != surface.shape
            or not 32 <= len(surface) <= 512 or not np.isfinite(surface).all()
            or not np.isfinite(normals).all()):
        raise ValueError('invalid surface/normal contract')
    query = result.get('query', surface[np.linspace(0, len(surface)-1, 32, dtype=int)])
    if query.shape != (32, 3) or not np.isfinite(query).all() or len(np.unique(query, axis=0)) != 32:
        raise ValueError('invalid fixed query identities')
    norms = np.linalg.norm(normals.astype(np.float64), axis=1)
    bad = norms <= policy.normal_min_norm
    if bad.any():
        touches_query = np.isclose(surface[bad, None], query[None], atol=1e-8, rtol=0).all(-1).any()
        if not policy.drop_nonquery_zero_normals or touches_query or (~bad).sum() < 32:
            raise ValueError('zero_or_tiny_normal_requires_mesh_review')
        result['surface'] = surface[~bad].copy()
    result['query'] = query.copy()
    result['normals'] = (normals[~bad] / norms[~bad, None]).astype(np.float32)
    return result, dict(points=len(surface), retained_points=int((~bad).sum()), dropped_nonquery_points=int(bad.sum()),
        original_norm_min=float(norms.min()),
        original_norm_max=float(norms.max()),
        rescaled_points=int((abs(norms[~bad]-1) > 1e-4).sum()),
        max_unit_error=float(abs(np.linalg.norm(result['normals'], axis=1)-1).max()),
        query_preserved=True, direction_preserved=True)


def canonical_sequence(values, row):
    """Decode compact simulator state separately from human native labels."""
    values = {k: np.asarray(v) for k, v in values.items()}
    if row['format'] == 'canonical':
        return values
    if row['format'] != 'unidexgrasp-compact':
        raise ValueError('unsupported native sequence format')
    if (str(values['object_code']) != row['object_code'] or float(values['scale']) != row['scale']
            or str(values['split']) != row['split'] or values['reset'].any()
            or values['training_allowed'].shape != ()
            or values['training_allowed'].dtype.kind != 'b' or bool(values['training_allowed'])):
        raise ValueError('compact identity/reset/admission mismatch')
    n = len(values['time'])
    if values['hand'].shape != (n, 11, 3):
        raise ValueError('compact hand shape mismatch')
    hand = np.zeros((n, 2, 11, 3), np.float32)
    hand[:, 0] = values['hand']
    valid = np.zeros((n, 2, 11), bool)
    valid[:, 0] = np.isfinite(hand[:, 0]).all(-1)
    return dict(time=values['time'], hand=hand, hand_valid=valid,
        object_poses=xyzquat_to_matrix(values['object'])[:, None],
        object_pose_valid=np.ones((n, 1), bool), segment=np.zeros(n, np.int64))


def clean_sequence(values, row, profile, policy):
    """Mask bad observations and mark clock boundaries without deleting time steps."""
    sequence = canonical_sequence(values, row)
    validate_sequence(sequence)
    n = len(sequence['time'])
    quality = np.ones(n, bool)
    boundary = np.zeros(n, bool)
    boundary[1:] = np.diff(sequence['segment']) != 0
    clock_verified = False
    report = dict(frames=n, native_gaps=0, repeated_or_backward_timestamps=0,
                  residual_rejected=0, native_quality_rejected=0)
    if row['format'] == 'unidexgrasp-compact':
        action_time = values['action_time']
        if (len(action_time) != 2*n-1 or not np.allclose(np.diff(action_time), 1/60, atol=1e-6, rtol=0)
                or not np.allclose(sequence['time'], action_time[::2], atol=1e-6, rtol=0)):
            raise ValueError('actual 60Hz action / 30Hz observed clock mismatch')
        clock_verified = True
    elif profile['clock_kind'] == 'timestamps':
        native, grid = sequence['native_timestamp_ns'], sequence['grid_timestamp_ns']
        residual = sequence['alignment_residual_ns']
        qa = sequence['native_qa_pass']
        if (native.shape != (n,) or grid.shape != (n,) or residual.shape != (n,)
                or native.dtype.kind not in 'iu' or grid.dtype.kind not in 'iu'
                or qa.shape != (n,) or qa.dtype.kind != 'b'
                or not np.array_equal(native-grid, residual)
                or not np.allclose(np.diff(grid)/1e9, 1/30, atol=1e-8, rtol=0)):
            raise ValueError('timestamp/residual/grid contract mismatch')
        nonincreasing = np.r_[False, np.diff(native) <= 0]
        gaps = np.r_[False, np.diff(native) > 1.5e9/30]
        quality &= qa & (abs(residual) <= policy.alignment_limit_ms*1e6) & ~nonincreasing
        boundary |= nonincreasing | gaps
        report.update(repeated_or_backward_timestamps=int(nonincreasing.sum()),
            native_gaps=int(gaps.sum()), native_quality_rejected=int((~qa).sum()),
            residual_rejected=int((abs(residual) > policy.alignment_limit_ms*1e6).sum()))
        clock_verified = True
    elif profile['clock_kind'] == 'resampled_fixed_fps':
        from .resampling import validate_resampled_clock
        validate_resampled_clock(sequence, profile['source_fps'])
        report.update(observed_frames=int(sequence['observed_frame'].sum()),
                      interpolated_frames=int(sequence['interpolated_frame'].sum()))
        clock_verified = True
    else:
        ids = sequence['native_frame_id']
        if ids.shape != (n,) or ids.dtype.kind not in 'iu':
            raise ValueError('explicit integer frame IDs required')
        jumps = np.r_[False, np.diff(ids) != 1]
        boundary |= jumps
        quality &= ~np.r_[False, np.diff(ids) <= 0]
        report['native_gaps'] = int(jumps.sum())
        clock_verified = profile['clock_kind'] == 'fixed_fps' and profile.get('source_fps') == 30
    cleaned = dict(sequence)
    cleaned['hand_valid'] = sequence['hand_valid'] & quality[:, None, None]
    cleaned['object_pose_valid'] = sequence['object_pose_valid'] & quality[:, None]
    cleaned['segment'] = np.cumsum(boundary).astype(np.int64)
    changed = any(not np.array_equal(cleaned[k], sequence[k])
                  for k in ('hand_valid', 'object_pose_valid', 'segment'))
    validate_sequence(cleaned)
    report.update(clock_verified=bool(clock_verified), quality_frames=int(quality.sum()),
                  changed_arrays=bool(changed), segments=int(cleaned['segment'][-1])+1)
    return cleaned, quality, report


def window_ticks(sequence, quality, target_slot, policy):
    """Require a complete target horizon and one sufficiently observed hand."""
    n = len(sequence['time'])
    if not 0 <= target_slot < sequence['object_poses'].shape[1]:
        raise ValueError('target slot out of range')
    ticks = np.arange(3, n-24, policy.stride, dtype=np.int64)
    reasons = {k: 0 for k in ('reset_or_clock_gap', 'frame_quality', 'target_history',
                             'target_future', 'hand_history', 'hand_future')}
    accepted = []
    for tick in ticks:
        whole = slice(tick-3, tick+25)
        history, future = slice(tick-3, tick+1), slice(tick+1, tick+25)
        present = sequence['hand_valid'][history].all(axis=(0, 2))
        coverage = sequence['hand_valid'][future].mean(axis=(0, 2))
        checks = ((sequence['segment'][whole] == sequence['segment'][tick]).all(),
            quality[whole].all(), sequence['object_pose_valid'][history, target_slot].all(),
            sequence['object_pose_valid'][future, target_slot].all(), present.any(),
            (present & (coverage >= policy.min_future_hand_fraction)).any())
        failure = next((k for k, ok in zip(reasons, checks) if not ok), None)
        if failure is None:
            accepted.append(tick)
        else:
            reasons[failure] += 1
    return np.asarray(accepted, np.int64), dict(candidates=len(ticks), accepted=len(accepted), rejected=reasons)


def reconcile_files(root, manifest, heldout=None):
    """A separate official TEST manifest owns its files; they are not orphans."""
    root = root.resolve()
    current = {(root/r['path']).resolve() for r in manifest['sequences']}
    test = {(root/r['path']).resolve() for r in (heldout or {}).get('sequences', [])}
    if current & test:
        raise ValueError('TRAIN/DEV overlaps heldout TEST paths')
    actual = {p.resolve() for p in (root/'sequences').glob('*.npz')}
    return dict(primary=len(current), heldout_test=len(test), disk=len(actual),
        missing=[str(p) for p in sorted((current | test)-actual)],
        unclaimed=[str(p) for p in sorted(actual-current-test)])
