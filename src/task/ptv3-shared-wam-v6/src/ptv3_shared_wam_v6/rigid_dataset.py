"""Rigid-only data admission and conservative content deduplication.

No object names, source IDs or model changes enter these predicates. Missing
annotations are distinct from explicit, correctly masked unknown annotations.
This is an audit/preparation seam; it does not loosen the production reader.
"""
import numpy as np

from ptv3_shared_wam_v5.contracts import JOINT_NAMES, PART_NAMES, LABEL_SEMANTICS


def admission_reasons(seq, geo, contact, event, evidence, tick, slot):
    reasons = []
    def reject(code):
        if code not in reasons:
            reasons.append(code)

    joints = seq.get('joints')
    n = len(joints) if joints is not None else evidence.get('frames', 0)
    if joints is None:
        reject('hand21_missing')
    elif joints.shape != (n, 2, 21, 3):
        reject('hand21_shape_invalid')
    if evidence.get('joint_semantics') != list(JOINT_NAMES):
        reject('hand21_semantics_unbound')
    if 'joint_valid' not in seq:
        reject('joint_valid_missing')
    elif seq['joint_valid'].shape != (n, 2, 21) or seq['joint_valid'].dtype != bool:
        reject('joint_valid_shape_or_dtype_invalid')
    poses = seq.get('object_poses')
    pose_valid = seq.get('object_pose_valid')
    pose_shape = poses is not None and poses.ndim == 4 and poses.shape[0] == n and poses.shape[2:] == (4, 4)
    if not pose_shape:
        reject('object_poses_missing_or_invalid')
    if pose_valid is None:
        reject('object_validity_missing')
    elif not pose_shape or pose_valid.shape != poses.shape[:2] or pose_valid.dtype != bool:
        reject('object_validity_shape_or_dtype_invalid')
    slots = poses.shape[1] if pose_shape else evidence.get('slots', 1)
    for name, shape in [('surface_offsets', (n, 2, 778, 3)), ('surface_valid', (n, 2)), ('surface_parts', (2, 778))]:
        if name not in seq or seq[name].shape != shape:
            reject('hand_surface_missing_or_invalid')
    if evidence.get('part_semantics') != list(PART_NAMES):
        reject('hand_part_semantics_unbound')
    for name, shape in [('surface', (2048, 3)), ('normals', (2048, 3)), ('query', (32, 3))]:
        if name not in geo:
            reject('surface_registry_missing')
        elif geo[name].shape != shape or not np.isfinite(geo[name]).all():
            reject('surface_registry_shape_or_values_invalid')
    if not evidence.get('geometry_binding_verified'):
        reject('surface_registry_unbound')
    if 'normals' in geo and geo['normals'].shape == (2048, 3):
        if not np.allclose(np.linalg.norm(geo['normals'], axis=-1), 1., atol=.01):
            reject('surface_normals_not_unit')
    for name, shape in [('contact', (n, slots, 2048)), ('part', (n, slots, 2048)), ('valid', (n, slots))]:
        if name not in contact:
            reject('contact_part_valid_missing')
        elif contact[name].shape != shape:
            reject('contact_shape_invalid')
    if evidence.get('label_semantics') != LABEL_SEMANTICS or not evidence.get('binding_verified'):
        reject('contact_semantics_unbound')
    for name in ('state', 'event_type', 'event_delta_steps', 'event_valid'):
        if name not in event:
            reject('event_missing')
        elif event[name].shape != (n, slots):
            reject('event_shape_invalid')
    if not evidence.get('event_binding_verified'):
        reject('event_contact_binding_unbound')
    if not 3 <= tick < n - 24 or not 0 <= slot < slots:
        reject('h4_k24_out_of_bounds')
    # Structural failures never enter indexing or silently clip a horizon.
    if any(x.endswith(('missing', 'invalid')) for x in reasons):
        return reasons
    w = slice(tick - 3, tick + 25)
    jv = seq['joint_valid'][w]
    if not jv[:4].all(0).any():
        reject('history_joint21_unobserved')
    if not np.isfinite(joints[w][jv]).all():
        reject('hand21_nonfinite_valid')
    pv = pose_valid[w, slot]
    if not pv.all():
        reject('object_pose_invalid_h4_k24')
    p = poses[w, slot]
    if not numerical_pose_valid(p).all():
        reject('object_pose_not_finite_se3')
    sv = seq['surface_valid'][w]
    if sv.dtype != bool or not np.isfinite(seq['surface_offsets'][w][sv]).all():
        reject('hand_surface_nonfinite_valid')
    if ((seq['surface_parts'] < 0) | (seq['surface_parts'] >= 16)).any():
        reject('hand_part_out_of_range')
    cv = contact['valid'][w, slot]
    if cv.dtype != bool:
        reject('contact_valid_not_bool')
    else:
        c = contact['contact'][w, slot][cv]
        parts = contact['part'][w, slot][cv]
        if not np.isfinite(c).all() or ((c < 0) | (c > 1)).any():
            reject('contact_confidence_invalid')
        if parts.dtype.kind not in 'iu' or ((parts < 0) | (parts >= 16)).any():
            reject('part_out_of_range')
    ev = event['event_valid'][w, slot]
    if ev.dtype != bool:
        reject('event_valid_not_bool')
    else:
        kind = event['event_type'][w, slot][ev]
        delta = event['event_delta_steps'][w, slot][ev]
        if not np.isin(kind, [0, 1, 2]).all() or (delta < 0).any():
            reject('event_type_or_time_invalid')
        # State may be unknown on an explicitly interpolated observation.
        if not np.isin(event['state'][w, slot], [-1, 0, 1]).all():
            reject('event_state_invalid')
    return reasons


def numerical_pose_valid(poses):
    """Numerical SE(3) gate only: not a substitute for annotation validity."""
    p = np.asarray(poses)
    r = p[..., :3, :3]
    return (np.isfinite(p).all(axis=(-2, -1))
            & np.isclose(p[..., 3, :], [0, 0, 0, 1], atol=1e-5).all(-1)
            & np.isclose(np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=1e-4).all(axis=(-2, -1))
            & np.isclose(np.linalg.det(r), 1., atol=1e-4))


def geometry_class(shape):
    """PCA bounding extents; deliberately shape classes, not object taxonomy."""
    s = np.sort(np.asarray(shape))[::-1]
    if not np.isfinite(s).all() or s[-1] <= 0:
        return 'unknown'
    if s[-1] / s[1] <= .2:
        return 'flat'
    if s[1] / s[0] <= .4:
        return 'elongated'
    return 'compact'


def pca_shape(points):
    p = np.asarray(points, np.float64)
    p = p - p.mean(0)
    _, vectors = np.linalg.eigh(p.T @ p)
    return np.sort(np.ptp(p @ vectors, axis=0))[::-1]


def duplicate_of(a, b):
    """Compare full H4/K24 content, never names or object/category identities.

    Shapes within5%; all anchored hand/query positions within2mm; identical
    validity and phase; each per-frame/per-part contact fraction within2%.
    Optional full object rotations also must agree within1degree. Comparison
    is to an actual retained representative, never transitive chaining.
    """
    if a['phase'] != b['phase'] or not np.array_equal(a['mask'], b['mask']):
        return False
    if np.max(np.abs(np.log(np.maximum(a['shape'], 1e-9) / np.maximum(b['shape'], 1e-9)))) > np.log(1.05):
        return False
    if np.max(np.abs(a['trajectory'] - b['trajectory'])) > .002:
        return False
    if np.max(np.abs(a['contact'] - b['contact'])) > .02:
        return False
    if 'rotations' in a:
        rel = np.swapaxes(a['rotations'], -1, -2) @ b['rotations']
        angle = np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
        if angle.max() > np.deg2rad(1):
            return False
    return True
