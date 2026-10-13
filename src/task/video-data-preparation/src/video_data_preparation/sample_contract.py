"""Validate a minimal forecasting sample without fake poses or actions."""
import numpy as np
from . import SOURCE_ID

FIXED_FIELDS = {
    "scene_points": ((1, 32, 3), "f"), "scene_target": ((1, 24, 32, 3), "f"),
    "scene_valid": ((1, 24, 32), "b"), "object_valid": ((1,), "b"),
    "start_object_points": ((32, 3), "f"), "goal_object_points": ((32, 3), "f"),
    "object_goal_valid": ((32,), "b"), "goal_object_slot": ((), "i"),
    "start_hand": ((2, 11, 3), "f"), "start_hand_history": ((2, 11, 9), "f"),
    "start_hand_valid": ((2, 11), "b"), "trajectory_target": ((24, 2, 11, 3), "f"),
    "trajectory_valid": ((24, 2, 11), "b"), "source_id": ((), "i"),
    "sample_id": ((2,), "i"),
}


def validate_sample(sample):
    n = len(sample["xyz"])
    if not 1 <= n <= 4096:
        raise ValueError("scene must contain 1..4096 non-hand points")
    expected = dict(FIXED_FIELDS, xyz=((n, 3), "f"), features=((n, 18), "f"),
                    scene_object=((n,), "i"))
    for key, (shape, kind) in expected.items():
        value = np.asarray(sample[key])
        if value.shape != shape or value.dtype.kind != kind:
            raise ValueError("invalid field " + key + ": expected " + str(shape))
        if kind == "f" and not np.isfinite(value).all():
            raise ValueError("nonfinite field " + key)
    if int(sample["source_id"]) != SOURCE_ID or int(sample["goal_object_slot"]) != 0:
        raise ValueError("EgoDex source=3 and current target slot=0 required")
    if not sample["object_valid"][0] or not (sample["features"][:, 14] == 0).all():
        raise ValueError("observed target and non-hand scene required")
    if not np.isin(sample["scene_object"], [0, -2]).all():
        raise ValueError("scene contains unknown object IDs")
    if not (sample["scene_object"] == 0).any():
        raise ValueError("target has no current scene support")
    if not np.array_equal(sample["scene_points"][0], sample["start_object_points"]):
        raise ValueError("scene and goal current point identities differ")
    mask = sample["object_goal_valid"]
    if not np.array_equal(mask, sample["scene_valid"][0, -1]):
        raise ValueError("goal mask must follow same endpoint track identities")
    delta = sample["goal_object_points"][mask] - sample["start_object_points"][mask]
    if not np.allclose(delta, sample["scene_target"][0, -1, mask], atol=2e-6):
        raise ValueError("endpoint goal and scene endpoint label differ")
    if (sample["trajectory_valid"] & ~sample["start_hand_valid"][None]).any():
        raise ValueError("future hand labels without observed hand")
    for key, mask in (("scene_target", sample["scene_valid"]),
                      ("goal_object_points", sample["object_goal_valid"]),
                      ("trajectory_target", sample["trajectory_valid"]),
                      ("start_hand", sample["start_hand_valid"]),
                      ("start_hand_history", sample["start_hand_valid"])):
        if np.any(sample[key][~mask] != 0):
            raise ValueError("invalid values must be zero: " + key)
    return sample
