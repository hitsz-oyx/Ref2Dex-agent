"""Offline contact diagnostics. Never imported by a model or training reader."""
import numpy as np

AUDIT_VERSION = 'v5.contact-prevalence-normal-candidate.v1'
THRESHOLDS = (0., .25, .5)
SOURCE_NAMES = {0: 'OakInk2', 1: 'GRAB', 2: 'ARCTIC', 4: 'Shadow/UniDex',
                5: 'TACO', 6: 'HOT3D', 8: 'DexYCB', 9: 'H2O', 10: 'HOI4D'}


def future_exposure(rows, frames, objects, horizon=24):
    """How often each cached frame/object is a target of the given windows."""
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError('slot,tick window rows required')
    slots, ticks = rows.astype(np.int64).T
    if ((slots < 0) | (slots >= objects) | (ticks < 3) | (ticks + horizon >= frames)).any():
        raise ValueError('window outside H4/K24 sequence')
    delta = np.zeros((frames + 1, objects), np.int64)
    np.add.at(delta, (ticks + 1, slots), 1)
    np.add.at(delta, (ticks + horizon + 1, slots), -1)
    return delta[:-1].cumsum(0)


def empty_counts():
    return dict(object_frames=0, queries=0, positive=[0, 0, 0], soft_mass=0.,
                frames_with_gt025=0, positive_part_counts=[0]*16)


def add_counts(total, other):
    for key in total:
        if isinstance(total[key], list):
            total[key] = [a+b for a, b in zip(total[key], other[key])]
        else:
            total[key] += other[key]


def contact_counts(values, valid, parts=None, exposure=None):
    """Streaming reducer; invalid geometry never enters the denominator.

    Exposure optionally weights a frame by its repeated K24 window appearances.
    This is uniform-window exposure, not the source/motion-stratum sampler.
    """
    values, valid = np.asarray(values), np.asarray(valid)
    if values.shape[:-1] != valid.shape or valid.dtype != np.bool_:
        raise ValueError('boolean frame/object validity must match contact shape')
    if exposure is None:
        exposure = np.ones(valid.shape, np.int64)
    exposure = np.asarray(exposure)
    if exposure.shape != valid.shape or not np.issubdtype(exposure.dtype, np.integer) or (exposure < 0).any():
        raise ValueError('nonnegative integer frame exposures required')
    mask = valid & (exposure > 0)
    x = values[mask]
    weight = exposure[mask]
    if not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError('valid contact confidence must be finite and in [0,1]')
    result = empty_counts()
    result['object_frames'] = int(weight.sum())
    result['queries'] = int(weight.sum())*values.shape[-1]
    result['positive'] = [int(((x > t).sum(-1)*weight).sum()) for t in THRESHOLDS]
    result['soft_mass'] = float((x.sum(-1, dtype=np.float64)*weight).sum())
    result['frames_with_gt025'] = int((np.any(x > .25, axis=-1)*weight).sum())
    if parts is not None:
        if np.shape(parts) != values.shape:
            raise ValueError('one semantic part per query required')
        labels = np.asarray(parts)[mask]
        positive = x > .25
        selected = labels[positive]
        if ((selected < 0) | (selected >= 16)).any():
            raise ValueError('canonical part outside [0,15]')
        weights = np.broadcast_to(weight[:, None], x.shape)[positive]
        result['positive_part_counts'] = np.bincount(selected.astype(int), weights=weights, minlength=16).astype(np.int64).tolist()
    return result


def describe_counts(counts):
    n = counts['queries']
    return dict(counts, prevalence={str(t): k/n if n else None for t, k in zip(THRESHOLDS, counts['positive'])},
                mean_soft_confidence=counts['soft_mass']/n if n else None)


def normal_candidate(vertices, vertex_parts, queries, normals, radius=.003,
                     falloff=.010, backside_tolerance=.001, backside_penalty=4.):
    """One explicit audit candidate, NOT an implementation of CEDex's formula.

    d_eff = d + 4 * max(-dot(h-o,n) - 1mm, 0).
    Use the existing 3mm/10mm/33mm confidence transfer on d_eff. The 1mm
    tolerance avoids penalizing small annotation penetration. Keep the same
    nearest vertex/part to isolate only the effect of object normal direction.
    Invalid normals are excluded from comparison and retain baseline confidence.
    """
    from scipy.spatial import cKDTree
    vertices, queries, normals = map(lambda x: np.asarray(x, np.float64), (vertices, queries, normals))
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or queries.ndim != 2 or queries.shape[1] != 3 or normals.shape != queries.shape
            or np.shape(vertex_parts) != (len(vertices),)):
        raise ValueError('3D vertices, per-vertex parts, queries and normals required')
    if not np.isfinite(vertices).all() or not np.isfinite(queries).all():
        raise ValueError('finite surface geometry required')
    if radius < 0 or falloff <= 0 or backside_tolerance < 0 or backside_penalty < 0:
        raise ValueError('invalid contact candidate scales')
    distance, nearest = cKDTree(vertices).query(queries, k=1)
    norm = np.linalg.norm(normals, axis=-1)
    valid = np.isfinite(normals).all(-1) & np.isfinite(norm) & (norm > 1e-8)
    unit = np.zeros_like(normals)
    unit[valid] = normals[valid]/norm[valid, None]
    residual = vertices[nearest]-queries
    signed = (residual*unit).sum(-1)
    effective = distance+backside_penalty*np.maximum(-signed-backside_tolerance, 0)
    effective[~valid] = distance[~valid]

    def confidence(d):
        c = np.exp(-np.maximum(d-radius, 0)/falloff)
        c[d > radius+3*falloff] = 0
        return c.astype(np.float16)

    return dict(baseline=confidence(distance), candidate=confidence(effective),
                distance=distance, signed=signed, normal_valid=valid,
                cosine=np.divide(signed, distance, out=np.zeros_like(signed), where=distance > 1e-12),
                nearest_part=np.asarray(vertex_parts)[nearest])


def opposite_surface_proxy(points, normals):
    """Nearby opposing samples are a thin-shell diagnostic, not a CAD proof."""
    from scipy.spatial import cKDTree
    points, normals = np.asarray(points), np.asarray(normals)
    norm = np.linalg.norm(normals, axis=-1)
    valid = np.isfinite(normals).all(-1) & (norm > 1e-8)
    unit = np.zeros_like(normals)
    unit[valid] = normals[valid]/norm[valid, None]
    d, ids = cKDTree(points).query(points, k=min(64, len(points)))
    if d.ndim == 1:
        return np.zeros(len(points), bool)
    opposed = (unit[:, None]*unit[ids]).sum(-1) < -.5
    residual = points[ids]-points[:, None]
    axial = np.abs((residual*unit[:, None]).sum(-1))
    return (opposed & valid[:, None] & valid[ids] & (d > 1e-6) & (d < .01) & (axial > .5*d)).any(-1)
