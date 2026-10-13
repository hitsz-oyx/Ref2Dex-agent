"""Turn one stored human clip into current inputs and relative targets.

Scene identities are chosen from the current valid mask only. Future motion
does not decide which points enter the model.
"""

from __future__ import annotations

import numpy as np


def select_scene_ids(current_valid: np.ndarray, max_points: int) -> np.ndarray:
    """Evenly keep up to ``max_points`` currently valid scene identities."""
    if max_points < 1:
        raise ValueError("max_points must be positive")
    chosen = np.flatnonzero(np.asarray(current_valid, dtype=bool))
    if len(chosen) > max_points:
        take = np.linspace(0, len(chosen) - 1, max_points).astype(np.int64)
        chosen = chosen[take]
    return chosen.astype(np.int64, copy=False)


def example_from_pack(pack: dict, max_scene_points: int) -> dict[str, np.ndarray]:
    """Sample one clip. Targets stay displacements from the current frame."""
    scene = np.asarray(pack["scene_xyz_m"], dtype=np.float32)
    scene_valid = np.asarray(pack["scene_valid"], dtype=bool)
    ids = select_scene_ids(scene_valid[0], max_scene_points)
    if len(ids) == 0:
        raise ValueError("clip has no currently valid scene point")
    hand = np.asarray(pack["hand_xyz_m"], dtype=np.float32)
    hand_valid = np.asarray(pack["hand_valid"], dtype=bool)
    return {
        "scene_xyz": scene[:, ids],
        "scene_rgb": np.asarray(pack["scene_rgb"], dtype=np.float32)[ids],
        "scene_valid": scene_valid[:, ids],
        "scene_displacement": np.asarray(pack["scene_displacement_m"], dtype=np.float32)[:, ids],
        "scene_target_valid": np.asarray(pack["scene_target_valid"], dtype=bool)[:, ids],
        "hand_xyz": hand,
        "hand_valid": hand_valid,
        "hand_displacement": np.asarray(pack["hand_displacement_m"], dtype=np.float32),
        "hand_target_valid": np.asarray(pack["hand_target_valid"], dtype=bool),
    }
