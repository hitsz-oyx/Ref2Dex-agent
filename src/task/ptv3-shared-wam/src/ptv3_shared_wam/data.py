"""Current-frame point forecasting labels assembled from existing SE(3)."""
import numpy as np
import torch

from .paths import add_dependencies

add_dependencies()
from goal_point_wam.data import GoalWindows, goal_collate
from oakink_wm.multisource import MixedWindows, MAIN_SOURCE_NAMES


def point_labels(sample, point_count=32):
    """All points and all future transforms use the SAME current anchor frame."""
    points = np.asarray(sample["points"], dtype=np.float32)
    effects = np.asarray(sample["effect"], dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3 or not 1 <= point_count <= points.shape[1]:
        raise ValueError("canonical object points and bounded point count required")
    if effects.shape != (len(points), 24, 4, 4):
        raise ValueError("object effects must be Ox24x4x4 in the current frame")
    ids = np.linspace(0, points.shape[1] - 1, point_count, dtype=np.int64)
    current = points[:, ids].copy()
    future = np.einsum("ohij,opj->ohpi", effects[..., :3, :3], current)
    future += effects[..., None, :3, 3]
    target = future - current[:, None]
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("finite current points and future labels required")
    return dict(scene_points=current, scene_target=target.astype(np.float32),
                scene_valid=np.ones(target.shape[:-1], dtype=bool))


class SharedWindows(torch.utils.data.Dataset):
    """Reuse the existing main-three splits and sampling, without rewriting data."""
    def __init__(self, root, split, point_count=32):
        self.base = MixedWindows(root, split)
        if tuple(s["name"] for s in self.base.meta["sources"]) != MAIN_SOURCE_NAMES:
            raise ValueError("only OakInk2/GRAB/ARCTIC admitted")
        self.goals = GoalWindows(self.base, "object_points", point_count)
        self.point_count = point_count

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = dict(self.goals[index])
        sample.update(point_labels(sample, self.point_count))
        return sample


def shared_collate(samples):
    batch = goal_collate(samples)
    count, objects = batch["object_valid"].shape
    points = samples[0]["scene_points"].shape[1]
    batch["scene_points"] = torch.zeros(count, objects, points, 3)
    batch["scene_target"] = torch.zeros(count, objects, 24, points, 3)
    batch["scene_valid"] = torch.zeros(count, objects, 24, points, dtype=torch.bool)
    for i, sample in enumerate(samples):
        if sample["scene_points"].shape[1] != points:
            raise ValueError("different per-object point counts in one batch")
        n = len(sample["scene_points"])
        for key in ("scene_points", "scene_target", "scene_valid"):
            batch[key][i, :n] = torch.as_tensor(sample[key])
    if "native_state" in samples[0]:
        for key in ("native_state", "native_tau", "native_action", "has_action"):
            batch[key] = torch.as_tensor(np.stack([sample[key] for sample in samples]))
    return batch


def observed_inputs(batch):
    """Forecaster allowlist excludes every future label and native robot state."""
    keys = ("xyz", "features", "point_valid", "scene_object", "object_valid",
            "scene_points", "start_hand", "start_hand_history", "start_hand_valid")
    return {key: batch[key] for key in keys}
