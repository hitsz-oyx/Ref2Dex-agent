"""Saved action statistics; command endpoints and measured joint aborts are distinct."""
import numpy as np

ACTION_TOLERANCE = 1e-6


def audit_action_boundaries(raw, action, q, limits, *, reported_clipped=None,
                            reported_source_rejections=None, source_chain=False):
    raw = np.asarray(raw, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if (raw.ndim != 2 or raw.shape[1] != 12 or action.shape != raw.shape
            or q.shape != (len(raw) + 1, 18)
            or not np.isfinite(np.r_[raw.ravel(), action.ravel(), q.ravel()]).all()):
        raise ValueError('finite T actions and T+1 measured joint states required')
    if not np.allclose(action, np.clip(raw, -1, 1), atol=1e-7, rtol=0):
        raise ValueError('saved action is not clip(raw)')
    changed = np.abs(raw - action) > ACTION_TOLERANCE
    clipped = changed.any(axis=1)
    endpoints = np.abs(action) >= 1 - ACTION_TOLERANCE
    strict_over = (np.abs(raw) > 1).any(axis=1)
    if reported_clipped is not None and int(reported_clipped) != int(clipped.sum()):
        raise ValueError('reported clipped_controls disagrees with raw trace')
    # A completed source rollout must contain no rejected command. NN chains do
    # not enter this branch: their zero source counter is not evidence of no clip.
    if source_chain and ((np.abs(raw) > 1 + ACTION_TOLERANCE).any()
                         or reported_source_rejections not in (None, 0)):
        raise ValueError('completed source chain contains rejected/saturated command')
    mask = np.asarray(limits['has_limits'], dtype=bool)
    lower = np.asarray(limits['lower'], dtype=np.float64)
    upper = np.asarray(limits['upper'], dtype=np.float64)
    if mask.shape != (18,) or lower.shape != (18,) or upper.shape != (18,):
        raise ValueError('18 native joint bounds required')
    nominal = mask & ((q < lower) | (q > upper))
    # Same numeric .10 as production: metres for 0:3, radians otherwise.
    abort = mask & ((q < lower - .10) | (q > upper + .10))
    return dict(controls=len(raw), clipped_controls=int(clipped.sum()),
                clipped_coordinates=int(changed.sum()), clipped_ticks=np.flatnonzero(clipped).tolist(),
                clipped_coordinates_per_action_axis=changed.sum(axis=0).tolist(),
                strict_raw_outside_controls=int(strict_over.sum()),
                boundary_controls=int(endpoints.any(axis=1).sum()),
                boundary_coordinates=int(endpoints.sum()),
                boundary_controls_cartesian_or_wrist=int(endpoints[:, :6].any(axis=1).sum()),
                boundary_ticks=np.flatnonzero(endpoints.any(axis=1)).tolist(),
                max_abs_raw=float(np.abs(raw).max()) if raw.size else 0.,
                max_clip_delta=float(np.abs(raw-action).max()) if raw.size else 0.,
                nominal_joint_outside_states=int(nominal.any(axis=1).sum()),
                joint_abort_states=int(abort.any(axis=1).sum()),
                first_joint_abort_tick=int(np.flatnonzero(abort.any(axis=1))[0]) if abort.any() else None,
                joint_abort_indices=np.flatnonzero(abort.any(axis=0)).tolist(),
                source_clipped_controls=int(reported_source_rejections or 0) if source_chain else None,
                source_counter_applicable=bool(source_chain))
