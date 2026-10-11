"""Critical geometry/missing-observation invariants, independent of models."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oakink_wm.pointwam_data import (ARKIT_TO_ZUP, JOINT_NAMES, lift_tracks,
    pack_clip, palm_keypoints, recover_metric_scale, resample_valid, select_scene)


def test_original_observation_recovers_after_invalid_gap():
    xyz = np.zeros((11, 1, 3), np.float32)
    xyz[:, 0, 0] = np.arange(11)
    valid = np.ones((11, 1), bool);valid[4] = False
    out, mask = resample_valid(xyz, valid)
    assert not mask[13, 0]  # between missing frame4 and observed frame5
    assert mask[15, 0] and out[15, 0, 0] == 5


def test_inf_and_nan_never_become_valid_targets():
    xyz = np.zeros((11, 1, 3), np.float32)
    xyz[4, 0, 0] = np.inf;xyz[8, 0, 1] = np.nan
    out, mask = resample_valid(xyz, np.ones((11, 1), bool))
    assert not mask[12, 0] and not mask[24, 0]
    assert np.isfinite(out[mask]).all()


def test_camera_translation_does_not_create_world_motion():
    # A world point fixed at (0,0,2); camera moves +.1m in x each step.
    camera = np.tile(np.eye(4, dtype=np.float32), (11, 1, 1))
    camera[:, 0, 3] = np.arange(11)*.1
    intrinsic = np.array([[10, 0, 10], [0, 10, 10], [0, 0, 1]], np.float32)
    uv = np.zeros((11, 1, 2), np.float32);uv[:, 0, 0] = 10-np.arange(11)*.5;uv[..., 1] = 10
    world, valid = lift_tracks(uv, np.ones((11, 1), bool), np.full((11, 24, 24), 2), intrinsic, camera)
    assert valid.all()
    np.testing.assert_allclose(world, np.broadcast_to([0, 0, 2], world.shape), atol=1e-6)
    np.testing.assert_allclose(ARKIT_TO_ZUP @ ARKIT_TO_ZUP.T, np.eye(3))
    assert np.linalg.det(ARKIT_TO_ZUP) == 1


def test_palm_offsets_are_rigid_and_tips_are_original_observations():
    joints = np.zeros((11, 2, len(JOINT_NAMES), 3), np.float32)
    joints[..., 1, 0] = .1;joints[..., 2, 1] = .04;joints[..., 3, 1] = -.04
    joints[..., 5:9, 0] = .2
    offsets = np.array([[0, 0, 0], [.02, 0, 0], [0, .02, 0],
                        [0, 0, .02], [-.02, 0, 0], [0, -.02, 0]], np.float32)
    out, mask = palm_keypoints(joints, np.ones(joints.shape[:-1], bool), offsets)
    np.testing.assert_allclose(out[..., 0, :], joints[..., 1, :])
    np.testing.assert_allclose(out[..., 6:, :], joints[..., 5:9, :])
    np.testing.assert_allclose(np.linalg.norm(out[..., 1:6, :]-out[..., :1, :], axis=-1), .02, atol=1e-6)
    assert mask.all()


def test_hand_scale_rejects_factor_two_outliers():
    camera = np.tile(np.eye(4, dtype=np.float32), (11, 1, 1))
    joints = np.zeros((11, 2, 9, 3), np.float32);joints[..., 2] = 2
    joints[0, 0, 0, 2] = 100
    result = recover_metric_scale(np.ones((11, 24, 24)), joints, np.ones(joints.shape[:-1], bool),
                                 np.array([[10, 0, 10], [0, 10, 10], [0, 0, 1]]), camera)
    assert result["scale"] == 2 and result["inliers"] == 197


def test_selection_never_depends_on_future_tracks_and_targets_have_30_steps():
    scene = np.zeros((11, 4, 3), np.float32);scene[:, :, 0] = np.arange(4)*.03
    valid = np.ones((11, 4), bool)
    chosen = select_scene(scene, valid)
    valid[1:] = False
    np.testing.assert_array_equal(chosen, select_scene(scene, valid))
    hand = np.zeros((11, 2, 10, 3), np.float32)
    pack = pack_clip(scene, valid, hand, np.ones(hand.shape[:-1], bool), np.zeros((4, 3), np.uint8),
                     np.arange(11)*3, "pick up")
    assert pack["scene_xyz_m"].shape[0] == 31
    assert pack["scene_displacement_m"].shape[0] == 30
    assert not pack["scene_target_valid"].any()
    assert pack["language"].item() == "pick up"
    assert not pack["training_allowed"].item()
