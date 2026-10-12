"""Adapter from the frozen three-source PointWorld samples to Goal-Point-WAM.

The source loader remains the authority for OakInk2/GRAB/ARCTIC splits and
normalization.  This module only exposes the extra endpoint-goal contract used
by the trajectory branch; it never reads future object labels into the scene
input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CM_SRC = Path(__file__).resolve().parents[2].parent / "cm-pointflow-effect-pretrain" / "src"
if str(CM_SRC) not in sys.path:
    sys.path.insert(0, str(CM_SRC))

from oakink_wm.data import collate as _base_collate  # noqa: E402


HORIZON = 24
HANDS = 2
KEYPOINTS = 11
ACTION_DIMS = 9

# The existing 18D scene contract is [normal, velocity, acceleration,
# history displacement, six metadata fields].  The final metadata fields are
# hand marker, hand id and normalized semantic keypoint id for current hand
# points.  Keep the indices explicit so this adapter cannot silently infer a
# hand layout from point order after object capping.
HAND_MARKER = 14
HAND_ID = 15
KEYPOINT_ID = 16
OBJECT_TARGET = 13


def _object_goal(sample: dict, point_count: int) -> dict:
    """Paired target-object endpoints in the single current anchor frame.

    The target flag identifies the anchor; never select an object using future
    motion or use the future object's own coordinate frame. Uniform canonical
    point indices are shared by both sets, without encoding their index as ID.
    """
    targets = np.flatnonzero(np.asarray(sample["object_features"])[:, OBJECT_TARGET] > .5)
    if len(targets) != 1:
        raise ValueError("object goal requires exactly one current target object")
    slot = int(targets[0])
    points = np.asarray(sample["points"], dtype=np.float32)[slot]
    if not 1 <= point_count <= len(points):
        raise ValueError("invalid object goal point count")
    indices = np.linspace(0, len(points) - 1, point_count, dtype=np.int64)
    start = points[indices].copy()
    effect = np.asarray(sample["effect"], dtype=np.float32)[slot, -1]
    goal = start @ effect[:3, :3].T + effect[:3, 3]
    if not np.isfinite(start).all() or not np.isfinite(goal).all():
        raise ValueError("nonfinite target-object endpoint")
    return dict(start_object_points=start, goal_object_points=goal.astype(np.float32),
                object_goal_valid=np.ones(point_count, dtype=bool),
                goal_object_slot=np.int64(slot))


def _current_hand(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    """Recover fixed (hand, keypoint) current points from the scene contract."""

    xyz = np.asarray(sample["xyz"], dtype=np.float32)
    features = np.asarray(sample["features"], dtype=np.float32)
    object_id = np.asarray(sample["scene_object"])
    if xyz.ndim != 2 or xyz.shape[1] != 3 or features.shape != (len(xyz), 18):
        raise ValueError("invalid current scene contract")
    points = np.zeros((HANDS, KEYPOINTS, 3), dtype=np.float32)
    valid = np.zeros((HANDS, KEYPOINTS), dtype=bool)
    rows = np.flatnonzero((object_id < 0) & (features[:, HAND_MARKER] > .5))
    for row in rows.tolist():
        hand = int(np.rint(features[row, HAND_ID]))
        keypoint = int(np.rint(features[row, KEYPOINT_ID] * 10.0))
        if not 0 <= hand < HANDS or not 0 <= keypoint < KEYPOINTS:
            raise ValueError("current hand metadata is outside the 2x11 contract")
        if valid[hand, keypoint]:
            raise ValueError("duplicate current hand semantic point")
        if not np.isfinite(xyz[row]).all():
            raise ValueError("current valid hand point is non-finite")
        points[hand, keypoint] = xyz[row]
        valid[hand, keypoint] = True
    # A source-level hand presence flag is deliberately not used to fabricate
    # missing keypoints.  The rows that actually exist are the validity mask.
    return points, valid


def _current_hand_history(sample: dict) -> np.ndarray:
    """Observed velocity/acceleration/three-step displacement, by hand ID."""
    # Validate the same semantic assignment used by current hand coordinates.
    _, valid = _current_hand(sample)
    features = np.asarray(sample['features'], dtype=np.float32)
    rows = np.flatnonzero((np.asarray(sample['scene_object']) < 0) & (features[:, HAND_MARKER] > .5))
    history = np.zeros((HANDS, KEYPOINTS, 9), dtype=np.float32)
    for row in rows:
        hand = int(np.rint(features[row, HAND_ID]))
        keypoint = int(np.rint(features[row, KEYPOINT_ID] * 10.))
        value = features[row, 3:12]
        if not np.isfinite(value).all():
            raise ValueError('nonfinite observed hand history')
        history[hand, keypoint] = value
    history[~valid] = 0.
    return history


class GoalWindows(Dataset):
    """Expose a frozen MixedWindows corpus with endpoint goal supervision."""

    def __init__(self, mixed_dataset, goal_type="hand_endpoint", goal_points=32):
        self.base = mixed_dataset
        self.source_names = tuple(d["name"] for d in mixed_dataset.meta["sources"])
        if goal_type not in ("hand_endpoint", "object_points"):
            raise ValueError("unknown goal contract")
        self.goal_type = goal_type
        self.goal_points = int(goal_points)

    def __len__(self) -> int:
        return len(self.base)

    def _source_id(self, index: int) -> int:
        return int(np.searchsorted(self.base.offsets, index, side="right") - 1)

    def __getitem__(self, index: int) -> dict:
        sample = dict(self.base[index])
        action = np.asarray(sample["action"], dtype=np.float32)
        action_valid = np.asarray(sample["action_valid"], dtype=bool)
        if action.shape != (HORIZON, HANDS * KEYPOINTS, ACTION_DIMS):
            raise ValueError(f"unexpected executed action shape: {action.shape}")
        if action_valid.shape != (HORIZON, HANDS * KEYPOINTS):
            raise ValueError(f"unexpected action mask shape: {action_valid.shape}")
        action = action.reshape(HORIZON, HANDS, KEYPOINTS, ACTION_DIMS)
        action_valid = action_valid.reshape(HORIZON, HANDS, KEYPOINTS)
        start, start_valid = _current_hand(sample)
        future = action[..., :3]
        # The trajectory target is relative to the observed current hand.  A
        # missing current point invalidates the corresponding future target;
        # no zero coordinate is ever admitted as supervision.
        trajectory_valid = action_valid & start_valid[None]
        trajectory_target = future - start[None]
        trajectory_target[~trajectory_valid] = 0.0
        goal_valid = action_valid[-1] & start_valid
        goal = np.where(goal_valid[..., None], future[-1], 0.0).astype(np.float32)
        sample.update(
            start_hand=start.astype(np.float32),
            start_hand_history=_current_hand_history(sample),
            start_hand_valid=start_valid,
            goal_hand=goal,
            goal_hand_valid=goal_valid,
            trajectory_target=trajectory_target.astype(np.float32),
            trajectory_valid=trajectory_valid,
            source_id=np.int64(self._source_id(index)),
        )
        if self.goal_type == "object_points":
            sample.update(_object_goal(sample, self.goal_points))
            # This contract does not expose future hand endpoint goal fields.
            del sample["goal_hand"], sample["goal_hand_valid"]
        return sample


def _cap_scene(samples: list[dict]) -> list[dict]:
    """Use the existing 4096-point object-preserving cap before base collate."""

    capped = []
    for original in samples:
        sample = dict(original)
        scene_object = np.asarray(sample["scene_object"])
        hand = np.flatnonzero(scene_object < 0)
        objects = len(sample["points"])
        if objects < 1:
            raise ValueError("scene has no supervised object")
        quota = min(512, (4096 - len(hand)) // objects)
        if quota < 1:
            raise ValueError("scene has too many hand/object points")
        take = [hand]
        for object_id in range(objects):
            pool = np.flatnonzero(scene_object == object_id)
            if not len(pool):
                raise ValueError("object has no scene points")
            take.append(pool[np.linspace(0, len(pool) - 1, quota, dtype=np.int64)])
        take = np.sort(np.concatenate(take))
        for key in ("xyz", "features", "scene_object"):
            sample[key] = sample[key][take]
        capped.append(sample)
    return capped


def goal_collate(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Collate variable object scenes plus the fixed goal/action contract."""

    if not samples:
        raise ValueError("cannot collate an empty sample list")
    capped = _cap_scene(samples)
    out = _base_collate(capped)
    max_points = out["point_valid"].shape[1]
    out["scene_object"] = torch.full((len(capped), max_points), -1, dtype=torch.long)
    for row, sample in enumerate(capped):
        out["scene_object"][row, : len(sample["scene_object"])] = torch.from_numpy(
            np.asarray(sample["scene_object"], dtype=np.int64)
        )
    for key, dtype in (
        ("start_hand", torch.float32),
        ("start_hand_history", torch.float32),
        ("trajectory_target", torch.float32),
        ("start_hand_valid", torch.bool),
        ("trajectory_valid", torch.bool),
        ("source_id", torch.long),
    ):
        out[key] = torch.as_tensor(np.stack([sample[key] for sample in capped]), dtype=dtype)
    object_goal = "goal_object_points" in capped[0]
    if any(("goal_object_points" in sample) != object_goal for sample in capped):
        raise ValueError("mixed goal contracts in one batch")
    fields = (("start_object_points", torch.float32), ("goal_object_points", torch.float32),
              ("object_goal_valid", torch.bool), ("goal_object_slot", torch.long)) if object_goal else (
              ("goal_hand", torch.float32), ("goal_hand_valid", torch.bool))
    for key, dtype in fields:
        out[key] = torch.as_tensor(np.stack([sample[key] for sample in capped]), dtype=dtype)
    return out


__all__ = ["GoalWindows", "goal_collate", "HORIZON", "HANDS", "KEYPOINTS"]
