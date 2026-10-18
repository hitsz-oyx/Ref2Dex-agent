"""WAM-v4 target-object surface and voxel interaction labels.

This module is deliberately NumPy-only and does not use KNN/cdist.  It turns
the canonical 11 hand points into a small semantic scaffold, rasterizes that
scaffold into a 1 cm integer grid, and labels fixed object query points by
voxel overlap.  The source surface/normal files remain read-only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
import numpy as np


# v3 requires distinct CAD surface identities as well as corrected C masks.
# v4 fixes the 18-D target/static feature contract and makes the C voxel
# origin depend only on the current observation.  Both change the admitted
# training input/label view, so the derived sample schema is versioned.
V4_SCHEMA = "ref2dex.wam-v4-target-object.v4"
FEATURE_TARGET = 12
FEATURE_STATIC = 13
STATIC_THRESHOLD_M = 0.002
INTERACTION_ORIGIN_SEMANTICS = "current-interaction-min-minus-5cm.v1"
SURFACE_POINTS = 512
FLOW_POINTS = 32
INTERACTION_POINTS = 128
VOXEL_M = 0.01
PART_NAMES = ("palm", "thumb", "index", "middle", "ring", "little")
PART_DIM = len(PART_NAMES)
C_DIM = 1 + PART_DIM

# The existing 11-point contract is MANO's wrist, five finger bases, then five
# finger tips: [wrist, thumb_mcp, index_mcp, middle_mcp, ring_mcp, little_mcp,
# thumb_tip, index_tip, middle_tip, ring_tip, little_tip].
SCAFFOLD_SEGMENTS = (
    (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 0), (0, 5, 0),
    (1, 6, 1), (2, 7, 2), (3, 8, 3), (4, 9, 4), (5, 10, 5),
)


@dataclass(frozen=True)
class V4SamplingPolicy:
    """Immutable policy for a derived v4 data view."""

    schema: str = V4_SCHEMA
    surface_points: int = SURFACE_POINTS
    flow_points: int = FLOW_POINTS
    interaction_points: int = INTERACTION_POINTS
    voxel_m: float = VOXEL_M
    dilation: int = 1
    static_keep_ratio: float = 0.10
    approach_keep_ratio: float = 0.35
    moving_object_keep_ratio: float = 0.35
    hand_motion_threshold_m: float = 0.002
    object_motion_threshold_m: float = 0.001

    def __post_init__(self):
        if self.schema != V4_SCHEMA or (self.surface_points, self.flow_points, self.interaction_points) != (512, 32, 128):
            raise ValueError("v4 uses the fixed 512/32/128 point contract")
        if not np.isfinite(self.voxel_m) or self.voxel_m <= 0 or self.dilation < 0:
            raise ValueError("positive voxel size and nonnegative dilation required")
        for value in (self.static_keep_ratio, self.approach_keep_ratio,
                      self.moving_object_keep_ratio):
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError("sampling ratios must be in (0,1]")
        for value in (self.hand_motion_threshold_m, self.object_motion_threshold_m):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("motion thresholds must be positive")


def fixed_subset(points: np.ndarray, count: int, *, name: str = "points") -> np.ndarray:
    """Select stable point identities without nearest-neighbour search."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < count:
        raise ValueError(f"{name} must contain at least {count} finite xyz points")
    if not np.isfinite(points).all():
        raise ValueError(f"nonfinite {name}")
    ids = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    selected = points[ids].astype(np.float32, copy=True)
    if len(np.unique(selected, axis=0)) != count:
        raise ValueError(f"{name} fixed subset is not unique")
    return selected


def resampled_subset(points: np.ndarray, count: int, *, name: str = "points") -> tuple[np.ndarray, np.ndarray]:
    """Select distinct canonical identities; underfilled geometry must be rebuilt."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1:
        raise ValueError(f"{name} must contain at least one finite xyz point")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite values")
    _, unique_ids = np.unique(points, axis=0, return_index=True)
    unique_ids.sort()
    if len(unique_ids) < count:
        raise ValueError(f"{name} needs {count} distinct CAD points; cyclic repeats are not allowed")
    ids = unique_ids[np.linspace(0, len(unique_ids) - 1, count, dtype=np.int64)]
    return points[ids].astype(np.float32, copy=True), ids


def _voxel(points: np.ndarray, origin: np.ndarray, voxel_m: float) -> np.ndarray:
    return np.floor((points - origin) / voxel_m).astype(np.int64)


def _scaffold(hand: np.ndarray, valid: np.ndarray, origin: np.ndarray, policy: V4SamplingPolicy):
    """Rasterize 2x11 hand points, returning voxel -> per-part counts."""
    points, parts = [], []
    for side in range(hand.shape[0]):
        for key, ok in enumerate(valid[side]):
            if ok:
                # The 11-point contract is wrist, five MCP points, then five
                # tips.  MCP/tip pairs share the same semantic finger id.
                # Keep this explicit: ``key - 1`` shifts index--little MCP
                # labels by one and silently corrupts the C supervision.
                part = 0 if key == 0 else (1 if key in (1, 6) else
                                           2 if key in (2, 7) else
                                           3 if key in (3, 8) else
                                           4 if key in (4, 9) else 5)
                points.append(hand[side, key])
                parts.append(part)
        for a, b, part in SCAFFOLD_SEGMENTS:
            if not (valid[side, a] and valid[side, b]):
                continue
            start, end = hand[side, a], hand[side, b]
            length = float(np.linalg.norm(end - start))
            n = max(2, int(np.ceil(length / policy.voxel_m)) + 1)
            line = start[None] + np.linspace(0, 1, n)[:, None] * (end - start)[None]
            points.append(line)
            parts.extend([part] * n)
    if not points:
        return np.empty((0, 3), np.int64), np.empty((0, PART_DIM), np.float32)
    voxels = _voxel(np.concatenate([p if p.ndim == 2 else p[None] for p in points]), origin, policy.voxel_m)
    unique, inverse = np.unique(voxels, axis=0, return_inverse=True)
    counts = np.zeros((len(unique), PART_DIM), np.float32)
    np.add.at(counts, (inverse, np.asarray(parts, np.int64)), 1.)
    return unique, counts


def _query_voxel_counts_reference(cells, cell_counts, query_voxels):
    """Original exact query path, retained for unsafe integer extents."""
    merged, inverse = np.unique(
        np.concatenate((cells, query_voxels.reshape(-1, 3))), axis=0, return_inverse=True)
    lookup = np.zeros((len(merged), PART_DIM), np.float32)
    lookup[inverse[:len(cells)]] = cell_counts
    return lookup[inverse[len(cells):]].reshape(*query_voxels.shape[:-1], PART_DIM).sum(1)


def _query_voxel_counts(cells, cell_counts, query_voxels):
    """Query unique occupied cells using collision-free bounded int64 keys.

    Sort only scaffold keys. Queries outside its bounding box are excluded
    before subtraction/encoding, including extreme signed coordinates. Python
    integer extent arithmetic detects overflow and retains the exact old path.
    Neighbour order and the float32 sum are unchanged.
    """
    if not len(cells):
        return np.zeros((len(query_voxels), PART_DIM), np.float32)
    lower, upper = cells.min(0), cells.max(0)
    spans = [int(hi) - int(lo) + 1 for lo, hi in zip(lower, upper)]
    volume = spans[0] * spans[1] * spans[2]
    if volume > np.iinfo(np.int64).max:
        return _query_voxel_counts_reference(cells, cell_counts, query_voxels)
    strides = np.array([spans[1] * spans[2], spans[2], 1], np.int64)
    keys = ((cells - lower) * strides).sum(-1)
    order = np.argsort(keys)
    keys = keys[order]
    flat = query_voxels.reshape(-1, 3)
    inside = ((flat >= lower) & (flat <= upper)).all(-1)
    ids = np.flatnonzero(inside)
    query_keys = ((flat[inside] - lower) * strides).sum(-1)
    positions = np.searchsorted(keys, query_keys)
    present = positions < len(keys)
    present[present] &= keys[positions[present]] == query_keys[present]
    found = np.zeros((len(flat), PART_DIM), np.float32)
    found[ids[present]] = cell_counts[order[positions[present]]]
    return found.reshape(*query_voxels.shape[:-1], PART_DIM).sum(1)


def scene_target_static(displacement: np.ndarray,
                        threshold: float = STATIC_THRESHOLD_M) -> np.ndarray:
    """Per-point H4 static flag, shared by every v4 reader and runtime.

    ``displacement`` is the current-minus-oldest point in the current object
    anchor frame.  A rigid target gives the same flag for every point; the
    per-point form keeps the contract explicit for future non-rigid views.
    """
    displacement = np.asarray(displacement, dtype=np.float32)
    if displacement.shape[-1] != 3 or not np.isfinite(displacement).all():
        raise ValueError("finite xyz H4 displacement required")
    return np.linalg.norm(displacement, axis=-1) < float(threshold)


def interaction_origin(current_points: np.ndarray) -> np.ndarray:
    """Voxel origin determined only by the current observed query geometry."""
    points = np.asarray(current_points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("finite current interaction points required")
    return points.min(0).astype(np.float64) - 0.05


def interaction_field(hand_future: np.ndarray, hand_valid: np.ndarray,
                      object_future: np.ndarray, policy: V4SamplingPolicy = V4SamplingPolicy(),
                      origin: np.ndarray | None = None,
                      active_sides: np.ndarray | None = None,
                      current_points: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``[T,Q,7]`` contact+hand-part labels and validity.

    ``object_future`` is the same fixed 128 object query identity at every
    frame.  Contact is exact voxel occupancy with one configurable dilation
    layer; part probabilities are normalized counts from the scaffold cells.
    """
    hand_future, hand_valid = np.asarray(hand_future), np.asarray(hand_valid)
    object_future = np.asarray(object_future)
    if hand_future.ndim != 4 or hand_future.shape[1:] != (2, 11, 3):
        raise ValueError("hand_future must be Tx2x11x3")
    if hand_valid.shape != hand_future.shape[:-1] or object_future.ndim != 3 or object_future.shape[0] != len(hand_future):
        raise ValueError("hand/object future shape mismatch")
    if object_future.shape[1:] != (policy.interaction_points, 3):
        raise ValueError("object_future must use 128 interaction queries")
    if not np.isfinite(hand_future[hand_valid]).all() or not np.isfinite(object_future).all():
        raise ValueError("finite valid hand/object future required")
    if origin is None:
        # The future t+1 frame must never choose the voxel phase for later
        # labels.  Callers pass current observed query geometry instead.
        if current_points is None:
            raise ValueError("interaction_field requires a current-observation origin")
        origin = interaction_origin(current_points)
    origin = np.asarray(origin, np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("finite xyz voxel origin required")
    target = np.zeros((len(hand_future), policy.interaction_points, C_DIM), np.float32)
    # A missing future hand frame means that contact is unknown, not negative.
    # Inactive hands are all-false throughout the window (normal single-hand
    # data), so only sides that are active anywhere are required to be fully
    # observed at a frame.  This preserves single-hand samples while masking
    # partial hand occlusion conservatively.
    future_active = hand_valid.any(axis=(0, 2))
    if active_sides is None:
        active_sides = future_active
    else:
        active_sides = np.asarray(active_sides, bool)
        if active_sides.shape != (2,):
            raise ValueError("active hand sides must have shape (2,)")
        # A hand observed at the current instant can disappear for the entire
        # future window. It must remain required for known-negative C labels.
        active_sides = active_sides | future_active
    frame_valid = np.zeros(len(hand_future), bool)
    if active_sides.any():
        frame_valid = hand_valid[:, active_sides].all(axis=(1, 2))
    valid = np.broadcast_to(frame_valid[:, None],
                            (len(hand_future), policy.interaction_points)).copy()
    offsets = np.asarray([(x, y, z) for x in range(-policy.dilation, policy.dilation + 1)
                          for y in range(-policy.dilation, policy.dilation + 1)
                          for z in range(-policy.dilation, policy.dilation + 1)], dtype=np.int64)
    for time, (hand, hvalid, obj) in enumerate(zip(hand_future, hand_valid, object_future)):
        if not frame_valid[time]:
            continue
        # Most approach/idle frames are far from the object.  An axis-aligned
        # conservative bound skips scaffold rasterization while preserving the
        # exact voxel result whenever contact is possible.
        visible = hand[hvalid]
        if len(visible):
            margin = policy.voxel_m * (policy.dilation + 1)
            if np.any(visible.max(0) + margin < obj.min(0)) or np.any(visible.min(0) - margin > obj.max(0)):
                continue
        cells, cell_counts = _scaffold(hand, hvalid, origin, policy)
        query_voxels = _voxel(obj, origin, policy.voxel_m)[:, None, :] + offsets[None, :, :]
        counts = _query_voxel_counts(cells, cell_counts, query_voxels)
        total = counts.sum(-1)
        contact = total > 0
        target[time, contact, 0] = 1.
        target[time, contact, 1:] = counts[contact] / total[contact, None]
    return target, valid


def classify_window(hand_future: np.ndarray, hand_valid: np.ndarray,
                    object_future: np.ndarray, target: np.ndarray,
                    policy: V4SamplingPolicy = V4SamplingPolicy()) -> str:
    """Classify a window for bounded static/approach sampling."""
    contact = target[..., 0].max(1) > 0
    hand_delta = np.linalg.norm(np.diff(hand_future, axis=0), axis=-1)
    object_delta = np.linalg.norm(np.diff(object_future, axis=0), axis=-1)
    hand_moving = bool(np.nanmax(np.where(hand_valid[1:], hand_delta, 0.)) >= policy.hand_motion_threshold_m)
    object_moving = bool(np.nanmax(object_delta) >= policy.object_motion_threshold_m)
    if contact.any():
        return "contact_moving" if object_moving else "contact_static"
    if hand_moving:
        return "approach"
    # No observed contact, hand still, object still moving: free fall, slide
    # or roll after release.  Calling this static_idle silently removed those
    # windows at the 10% idle rate.
    if object_moving:
        return "moving_object_noncontact"
    return "static_idle"


def retain_window(kind: str, identity: object, policy: V4SamplingPolicy = V4SamplingPolicy()) -> bool:
    """Deterministically thin idle and noncontact approach windows."""
    if kind == "contact_moving" or kind == "contact_static":
        return True
    digest = hashlib.sha256(repr(identity).encode()).digest()
    draw = int.from_bytes(digest[:8], "little") / float(2**64)
    ratios = {
        "approach": policy.approach_keep_ratio,
        "moving_object_noncontact": policy.moving_object_keep_ratio,
        "static_idle": policy.static_keep_ratio,
    }
    if kind not in ratios:
        raise ValueError("unknown v4 window kind: " + str(kind))
    return draw < ratios[kind]
