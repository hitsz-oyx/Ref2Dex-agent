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


# v2 records the corrected active-hand validity mask and semantic finger map.
V4_SCHEMA = "ref2dex.wam-v4-target-object.v2"
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
    hand_motion_threshold_m: float = 0.002
    object_motion_threshold_m: float = 0.001

    def __post_init__(self):
        if self.schema != V4_SCHEMA or (self.surface_points, self.flow_points, self.interaction_points) != (512, 32, 128):
            raise ValueError("v4 uses the fixed 512/32/128 point contract")
        if not np.isfinite(self.voxel_m) or self.voxel_m <= 0 or self.dilation < 0:
            raise ValueError("positive voxel size and nonnegative dilation required")
        for value in (self.static_keep_ratio, self.approach_keep_ratio):
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
    """Return a deterministic fixed-size view and the source indices.

    Native legacy geometry can contain fewer than 512 surface samples.  The
    v4 contract keeps those samples and repeats their canonical identities in
    a cyclic, auditable way; it never invents a normal or perturbs a point.
    Dense main3 geometry uses the one-to-one ``fixed_subset`` path.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1:
        raise ValueError(f"{name} must contain at least one finite xyz point")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite values")
    if len(points) >= count:
        ids = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    else:
        ids = np.arange(count, dtype=np.int64) % len(points)
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


def interaction_field(hand_future: np.ndarray, hand_valid: np.ndarray,
                      object_future: np.ndarray, policy: V4SamplingPolicy = V4SamplingPolicy(),
                      origin: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
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
        origin = object_future[0].min(0) - 0.05
    origin = np.asarray(origin, np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("finite xyz voxel origin required")
    target = np.zeros((len(hand_future), policy.interaction_points, C_DIM), np.float32)
    # A missing future hand frame means that contact is unknown, not negative.
    # Inactive hands are all-false throughout the window (normal single-hand
    # data), so only sides that are active anywhere are required to be fully
    # observed at a frame.  This preserves single-hand samples while masking
    # partial hand occlusion conservatively.
    active_sides = hand_valid.any(axis=(0, 2))
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
        # Merge scaffold and all query-neighbour voxels once, then gather the
        # scaffold counts by inverse ID.  This preserves exact integer voxel
        # occupancy while avoiding Q*27 Python dictionary lookups per frame.
        merged, inverse = np.unique(
            np.concatenate((cells, query_voxels.reshape(-1, 3)), axis=0),
            axis=0, return_inverse=True)
        lookup = np.zeros((len(merged), PART_DIM), np.float32)
        lookup[inverse[:len(cells)]] = cell_counts
        counts = lookup[inverse[len(cells):]].reshape(len(obj), len(offsets), PART_DIM).sum(1)
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
    return "approach" if hand_moving else "static_idle"


def retain_window(kind: str, identity: object, policy: V4SamplingPolicy = V4SamplingPolicy()) -> bool:
    """Deterministically thin idle and noncontact approach windows."""
    if kind == "contact_moving" or kind == "contact_static":
        return True
    digest = hashlib.sha256(repr(identity).encode()).digest()
    draw = int.from_bytes(digest[:8], "little") / float(2**64)
    ratio = policy.approach_keep_ratio if kind == "approach" else policy.static_keep_ratio
    return draw < ratio
