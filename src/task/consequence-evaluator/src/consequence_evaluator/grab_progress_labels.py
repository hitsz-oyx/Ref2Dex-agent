"""Versioned weak stage boundaries using native contact and per-frame support planes."""
import numpy as np

DEFAULTS = dict(contact_frames=3, leave_clearance_m=.01, target_clearance_m=.03,
                hold_frames=15, approach_displacement_m=.01,
                return_descent_m=.02, return_frames=3)


def first_run(mask, length, start=0):
    count = 0
    for i in range(start, len(mask)):
        count = count+1 if mask[i] else 0
        if count >= length:
            return i-length+1
    return None


def propose(clearance, right_contact, left_contact, other_contact, wrist, config=None):
    c = dict(DEFAULTS, **(config or {}))
    n = len(clearance)
    if any(len(x) != n for x in (right_contact, left_contact, other_contact, wrist)):
        raise ValueError('unaligned label signals')
    b1 = first_run(right_contact, c['contact_frames'])
    if b1 is None or b1 < 2:
        return None, 'no_right_contact_after_approach'
    b2 = first_run(right_contact & (clearance > c['leave_clearance_m']), c['contact_frames'], b1+1)
    if b2 is None:
        return None, 'no_supported_lift'
    b3 = first_run(right_contact & (clearance > c['target_clearance_m']), c['hold_frames'], b2+1)
    if b3 is None or b3 <= b2:
        return None, 'no_verified_hold'
    moving = np.linalg.norm(wrist-wrist[0], axis=-1) > c['approach_displacement_m']
    b0 = first_run(moving, c['contact_frames'])
    b0 = 0 if b0 is None else max(0, min(b0-1, b1-1))
    end = n
    # Stop at evidence of active lowering, returning to support or releasing.
    peak = np.maximum.accumulate(clearance)
    returning = ((peak-clearance > c['return_descent_m'])
                 & (np.r_[0., np.diff(clearance)] < 0))
    for mask in (returning, clearance < c['target_clearance_m'], ~right_contact):
        tick = first_run(mask, c['return_frames'], b3+c['hold_frames'])
        if tick is not None:
            end = min(end, tick)
    if end < b3+c['hold_frames']:
        return None, 'hold_truncated'
    if first_run(left_contact | other_contact, c['contact_frames'], b1) is not None:
        # Only the retained grasp counts; later placing contacts do not contaminate it.
        if (left_contact[b1:end] | other_contact[b1:end]).any():
            return None, 'other_hand_or_body_support'
    if not b0 < b1 < b2 < b3 < end:
        return None, 'ambiguous_stage_order'
    return dict(boundaries=[int(b0), int(b1), int(b2), int(b3)], end=int(end)), None


def training_stage_weights(boundaries):
    durations = np.diff(np.asarray(boundaries, dtype=float), axis=1)
    if durations.ndim != 2 or durations.shape[1] != 3 or (durations <= 0).any():
        raise ValueError('ordered training stage boundaries required')
    # SARM: mean of per-sequence duration proportions; hold duration is excluded.
    return (durations/durations.sum(1, keepdims=True)).mean(0)


def dense_labels(n, boundaries, end, alpha):
    b = list(map(int, boundaries))
    a = np.asarray(alpha, dtype=float)
    if not len(b) == 4 or not 0 <= b[0] < b[1] < b[2] < b[3] < end <= n:
        raise ValueError('ordered complete stage boundaries required')
    if a.shape != (3,) or (a <= 0).any() or not np.isclose(a.sum(), 1):
        raise ValueError('three train-only stage proportions required')
    stage = np.full(n, -1, np.int64)
    progress = np.zeros(n, np.float32)
    valid = np.zeros(n, bool)
    for s in range(3):
        ids = np.arange(b[s], b[s+1])
        stage[ids] = s
        progress[ids] = a[:s].sum()+a[s]*(ids-b[s])/(b[s+1]-b[s])
    stage[b[3]:end] = 3
    progress[b[3]:end] = 1
    valid[b[0]:end] = True
    return stage, progress, valid


def support_planes(vertices, rotations, translations):
    """Top face of the thin table mesh; rotations are column-vector local→world."""
    v = np.asarray(vertices, float)
    extent = np.ptp(v, axis=0)
    axis = int(np.argmin(extent))
    if extent[axis] > .05*np.max(extent):
        raise ValueError('table mesh is not an unambiguous thin support slab')
    normal = np.eye(3)[axis]
    if (rotations[0]@normal)[2] < 0:
        normal = -normal
    normals = np.einsum('tij,j->ti', rotations, normal)
    if (normals[:, 2] < .9).any():
        raise ValueError('table plane orientation inconsistent with world up')
    top = float(np.max(v@normal))
    offsets = -np.einsum('ti,ti->t', normals, translations)-top
    return np.concatenate((normals, offsets[:, None]), axis=1).astype('float32')


def plane_in_object_frame(planes, poses):
    normals = np.einsum('tji,tj->ti', poses[:, :3, :3], planes[:, :3])
    offset = np.einsum('ti,ti->t', planes[:, :3], poses[:, :3, 3])+planes[:, 3]
    return np.concatenate((normals, offset[:, None]), axis=1).astype('float32')


def lowest_clearance(vertices, plane_object):
    values = []
    for start in range(0, len(plane_object), 32):
        p = plane_object[start:start+32]
        values.extend((np.min(p[:, :3]@vertices.T, axis=1)+p[:, 3]).tolist())
    return np.asarray(values, np.float32)
