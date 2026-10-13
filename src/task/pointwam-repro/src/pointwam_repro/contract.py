"""Checks for the stored PointWAM human-clip schema.

The packs are teacher reconstructions. Displacement is relative to the current
frame, not a step-to-step difference. ``training_allowed`` stays false until a
later probe explicitly admits a split.
"""

from __future__ import annotations

import numpy as np

SCHEMA = "ref2dex.pointwam-data-smoke.v1"
ARKIT_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
REQUIRED = (
    "scene_xyz_m",
    "scene_valid",
    "scene_rgb",
    "hand_xyz_m",
    "hand_valid",
    "scene_displacement_m",
    "hand_displacement_m",
    "scene_target_valid",
    "hand_target_valid",
    "source_frame_ids",
    "timestamps_s",
    "arkit_to_zup",
    "training_allowed",
    "schema",
)


def audit_pack(pack: dict) -> dict[str, object]:
    """Return whether one loaded clip matches the verified human contract."""
    errors: list[str] = []
    missing = [key for key in REQUIRED if key not in pack]
    if missing:
        return {"passed": False, "errors": ["missing:" + ",".join(missing)]}
    scene = np.asarray(pack["scene_xyz_m"])
    scene_valid = np.asarray(pack["scene_valid"], dtype=bool)
    hand = np.asarray(pack["hand_xyz_m"])
    hand_valid = np.asarray(pack["hand_valid"], dtype=bool)
    scene_delta = np.asarray(pack["scene_displacement_m"])
    hand_delta = np.asarray(pack["hand_displacement_m"])
    scene_target = np.asarray(pack["scene_target_valid"], dtype=bool)
    hand_target = np.asarray(pack["hand_target_valid"], dtype=bool)
    if scene.ndim != 3 or scene.shape[0] != 31 or scene.shape[-1] != 3:
        errors.append(f"scene_shape {scene.shape}")
    elif scene_delta.shape != (30,) + scene.shape[1:]:
        errors.append(f"scene_displacement_shape {scene_delta.shape}")
    if hand.shape != (31, 2, 10, 3) or hand_delta.shape != (30, 2, 10, 3):
        errors.append(f"hand_shape {hand.shape} {hand_delta.shape}")
    if scene_valid.shape != scene.shape[:-1] or hand_valid.shape != hand.shape[:-1]:
        errors.append("mask_shape")
    if scene_target.shape != scene_delta.shape[:-1] or hand_target.shape != hand_delta.shape[:-1]:
        errors.append("target_mask_shape")
    clocks = np.asarray(pack["timestamps_s"], dtype=np.float64)
    if clocks.shape != (31,) or not np.all(np.diff(clocks) > 0):
        errors.append("clock")
    frames = np.asarray(pack["source_frame_ids"])
    if frames.shape != (11,) or not np.array_equal(np.diff(frames), np.full(10, 3)):
        errors.append("source_ids")
    if not np.allclose(np.asarray(pack["arkit_to_zup"], dtype=np.float32), ARKIT_TO_ZUP):
        errors.append("zup")
    if str(np.asarray(pack["schema"]).reshape(-1)[0]) != SCHEMA:
        errors.append("schema")
    if bool(np.asarray(pack["training_allowed"]).reshape(-1)[0]):
        errors.append("training_allowed")
    if scene_valid.any() and not np.isfinite(scene[scene_valid]).all():
        errors.append("nonfinite_scene")
    if hand_valid.any() and not np.isfinite(hand[hand_valid]).all():
        errors.append("nonfinite_hand")
    if scene.shape[:1] == (31,) and scene_delta.shape[0] == 30:
        compared = scene_target & scene_valid[0]
        if compared.any():
            gap = np.abs((scene[1:] - scene[0]) - scene_delta)[compared]
            if float(gap.max()) > 1e-5:
                errors.append("scene_displacement")
    if hand.shape == (31, 2, 10, 3):
        compared = hand_target & hand_valid[0]
        if compared.any():
            gap = np.abs((hand[1:] - hand[0]) - hand_delta)[compared]
            if float(gap.max()) > 1e-5:
                errors.append("hand_displacement")
        if not bool(hand_valid.all(axis=(0, 2)).any()):
            errors.append("no_complete_hand")
    points = int(scene.shape[1]) if scene.ndim == 3 else 0
    if points < 256 or points > 12000:
        errors.append(f"npoints {points}")
    return {"passed": not errors, "errors": errors, "scene_points": points}
