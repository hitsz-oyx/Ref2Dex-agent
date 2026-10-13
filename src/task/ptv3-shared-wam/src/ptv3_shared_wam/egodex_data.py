"""EgoDex provider and minimal mixed-source collate, isolated from active runs."""
from pathlib import Path
import sys
import numpy as np
import torch

from .data import endpoint_delta
from .paths import TASK

sys.path.insert(0, str(TASK.parent / "video-data-preparation" / "src"))
from video_data_preparation.shards import ShardDataset


class EgoDexSharedWindows(ShardDataset, torch.utils.data.Dataset):
    pass


def forecasting_collate(samples, scene_cap=4096):
    """No SE(3), action or canonical-point placeholders required by this path.

    Legacy providers retain their existing selection before the neutral boundary;
    EgoDex keeps target queries, local support and context chosen at preprocessing.
    """
    if not samples or not 32 <= scene_cap <= 4096:
        raise ValueError("nonempty forecasting batch and bounded cap required")
    from goal_point_wam.data import _cap_scene
    selected = [_cap_scene([s])[0] if "points" in s else s for s in samples]
    count = len(samples)
    sizes = [min(len(s["xyz"]), scene_cap) for s in selected]
    objects = max(len(s["scene_points"]) for s in selected)
    batch = dict(xyz=torch.zeros(count, max(sizes), 3), features=torch.zeros(count, max(sizes), 18),
                 point_valid=torch.zeros(count, max(sizes), dtype=torch.bool),
                 scene_object=torch.full((count, max(sizes)), -2, dtype=torch.long),
                 object_valid=torch.zeros(count, objects, dtype=torch.bool),
                 scene_points=torch.zeros(count, objects, 32, 3),
                 scene_target=torch.zeros(count, objects, 24, 32, 3),
                 scene_valid=torch.zeros(count, objects, 24, 32, dtype=torch.bool))
    for row, (sample, size) in enumerate(zip(selected, sizes)):
        if sample["scene_points"].shape[1:] != (32, 3):
            raise ValueError("fixed 32 same-ID queries per object required")
        n = len(sample["scene_points"])
        for key in ("xyz", "features", "scene_object"):
            value = np.asarray(sample[key])
            if len(value) != len(sample["xyz"]):
                raise ValueError("scene geometry/feature/ID lengths differ")
            batch[key][row, :size] = torch.as_tensor(value[:size])
        if size < 1 or not np.isfinite(sample["xyz"][:size]).all():
            raise ValueError("finite nonempty current scene required")
        batch["point_valid"][row, :size] = True
        batch["object_valid"][row, :n] = True
        for key in ("scene_points", "scene_target", "scene_valid"):
            batch[key][row, :n] = torch.as_tensor(sample[key])
    fixed = ("start_hand", "start_hand_history", "start_hand_valid", "trajectory_target", "trajectory_valid",
             "start_object_points", "goal_object_points", "object_goal_valid", "goal_object_slot", "source_id")
    for key in fixed:
        batch[key] = torch.as_tensor(np.stack([s[key] for s in selected]))
    batch["goal_object_delta"] = endpoint_delta(batch["start_object_points"], batch["goal_object_points"], batch["object_goal_valid"])
    # Legacy sample_id has a different tuple size. Only source_id is a model metric field.
    if len({np.shape(s["sample_id"]) for s in selected}) == 1:
        batch["sample_id"] = torch.as_tensor(np.stack([s["sample_id"] for s in selected]))
    if any("native_state" in s for s in samples):
        raise ValueError("EgoDex/mixed forecasting path cannot fabricate robot action batches")
    return batch
