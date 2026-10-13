"""Camera/identity/causality checks on actual window construction, not model mocks."""
import copy
import json
import numpy as np
import pytest
import h5py

from video_data_preparation.geometry import lift_tracks, sample_depth, resize_intrinsic
from video_data_preparation.native_hand import NativeEpisode, JOINT_SUFFIXES, read_native
from video_data_preparation.pipeline import ProcessingConfig, RejectedWindow, process_window


def fixture_episode(session="session-train"):
    frames = 40
    k = np.array([[40., 0., 24.], [0., 40., 16.], [0., 0., 1.]], np.float32)
    camera = np.repeat(np.eye(4, dtype=np.float32)[None], frames, 0)
    hand = np.zeros((frames, 2, 11, 3), np.float32)
    hand[..., 2] = 1.
    hand[..., 0] = np.linspace(-.2, .2, 11)
    hand[..., 1] = np.array([-.05, .05])[None, :, None]
    valid = np.ones((frames, 2, 11), bool)
    return NativeEpisode(hand, valid, camera, k, session, "move the block")


class Masks:
    def select_current(self, key, tick, rgb, native):
        assert len(native.hand) == 4  # Future native arrays cannot enter selector.
        target = np.zeros(rgb.shape[:2], bool)
        target[5:27, 7:41] = True
        return dict(target=target, hand=np.zeros_like(target), name="block", current_verified=True)

    def label_masks(self, key, frame_ids, shape):
        return np.ones((len(frame_ids), *shape), bool)


class Depth:
    def predict(self, rgb, frame_ids, role):
        assert len(rgb) == (4 if role == "observation" else 25)
        return np.ones(rgb.shape[:3], np.float32)


class Tracker:
    def __init__(self, occluded=False, reversed_ids=False):
        self.occluded, self.reversed_ids = occluded, reversed_ids

    def track(self, rgb, uv):
        tracks = np.repeat(uv[None], 25, 0)
        if self.reversed_ids:
            tracks = tracks[:, ::-1].copy()
        visible = np.ones(tracks.shape[:-1], bool)
        if self.occluded:
            visible[8:11, :3] = False
            visible[-1, :4] = False
        return tracks, visible


def built(native=None, **kwargs):
    native = native or fixture_episode()
    return process_window(native, "part2/pick/0.hdf5", 7, 3,
        np.zeros((28, 32, 48, 3), np.uint8), native.intrinsic,
        kwargs.get("masks", Masks()), kwargs.get("depth", Depth()),
        kwargs.get("tracker", Tracker()), ProcessingConfig(scene_points=128, erosion=1))


def test_native_mapping_history_and_missing_confidence(tmp_path):
    p = tmp_path/"native.hdf5"
    episode = fixture_episode()
    with h5py.File(p, "w") as f:
        f["transforms/camera"] = episode.camera
        f["camera/intrinsic"] = episode.intrinsic
        f.attrs["session_name"] = "recording.mov"
        f.attrs["which_llm_description"] = 2
        f.attrs["llm_description2"] = "place the block"
        for side in ("right", "left"):
            for j, suffix in enumerate(JOINT_SUFFIXES):
                values = episode.camera.copy()
                values[:, 0, 3] = np.arange(40)*.001+j*.01
                values[:, 2, 3] = 1.
                f["transforms/"+side+suffix] = values
                f["confidences/"+side+suffix] = np.ones(40, np.float32)
    loaded = read_native(p)
    assert loaded.language == "place the block"
    result = loaded.window(3, np.eye(4))
    np.testing.assert_allclose(result["start_hand_history"][..., 0], .03, atol=1e-6)
    np.testing.assert_allclose(result["trajectory_target"][-1, ..., 0], .024, atol=1e-6)
    loaded.valid[1, 0, 6] = False
    result = loaded.window(3, np.eye(4))
    assert not result["start_hand_valid"][0, 6]
    assert not result["trajectory_valid"][:, 0, 6].any()
    assert (result["trajectory_target"][:, 0, 6] == 0).all()
    with h5py.File(p, "a") as f:
        del f["confidences"]
    with pytest.raises(ValueError, match="missing native confidences"):
        read_native(p)


def test_static_point_stays_static_under_moving_camera():
    k = np.array([[40., 0., 24.], [0., 40., 16.], [0., 0., 1.]], np.float32)
    camera = np.repeat(np.eye(4)[None], 25, 0)
    camera[:, 0, 3] = np.arange(25)*.005
    uv = np.zeros((25, 1, 2), np.float32)
    uv[..., 0] = 24.-40.*camera[:, 0, 3, None]
    uv[..., 1] = 16.
    xyz, good = lift_tracks(uv, np.ones((25, 1), bool), np.ones((25, 32, 48)), k, camera)
    assert good.all()
    np.testing.assert_allclose(xyz, np.broadcast_to([0, 0, 1], xyz.shape), atol=1e-6)


def test_observation_future_isolation_and_context():
    native = fixture_episode()
    a, meta, _ = built(native)
    changed = copy.deepcopy(native)
    # Labels must change, while current inputs are byte-identical.
    changed.hand[4:, ..., 0] += .002*np.arange(1, 37)[:, None, None]
    b, _, _ = built(changed)
    for key in ("xyz", "features", "scene_points", "scene_object", "start_hand",
                "start_hand_history", "start_hand_valid", "start_object_points"):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["trajectory_target"], b["trajectory_target"])
    assert (a["scene_object"] == -2).any()
    assert len(a["xyz"]) <= 128
    assert meta["observation_frame_ids"] == [0, 1, 2, 3]
    assert meta["label_frame_ids"] == list(range(3, 28))
    assert meta["causal_observation"]
    assert (a["features"][:, 3:12] == 0).all()
    assert (a["features"][:, 14] == 0).all()


def test_occlusion_keeps_ids_and_masks_labels():
    clean, _, _ = built()
    masked, meta, audit = built(tracker=Tracker(occluded=True))
    np.testing.assert_array_equal(clean["scene_points"], masked["scene_points"])
    np.testing.assert_array_equal(clean["xyz"], masked["xyz"])
    assert not masked["scene_valid"][0, 7:10, :3].any()
    assert (masked["scene_target"][0, 7:10, :3] == 0).all()
    assert not masked["object_goal_valid"][:4].any()
    assert (masked["goal_object_points"][:4] == 0).all()
    assert meta["endpoint_fraction"] == 28/32
    assert not audit["track_valid"][-1, :4].any()


def test_reordered_tracks_and_missing_identity_rejected():
    with pytest.raises(ValueError, match="changed current query"):
        built(tracker=Tracker(reversed_ids=True))
    class NoIdentity(Masks):
        def label_masks(self, *args):
            return None
    with pytest.raises(RejectedWindow, match="MISSING_FUTURE"):
        built(masks=NoIdentity())


def test_invalid_depth_and_resized_pixel_centers():
    depth = np.ones((1, 3, 3), np.float32)
    depth[0, 1, 1] = np.nan
    z, mask = sample_depth(depth, np.array([[[.5, .5], [2., 2.], [np.nan, 1.]]]))
    assert not mask.any()
    assert (z == 0).all()
    k = resize_intrinsic(np.array([[40., 0., 24.], [0., 40., 16.], [0., 0., 1.]]), (48, 32), (24, 16))
    np.testing.assert_allclose(k, [[20, 0, 11.75], [0, 20, 7.75], [0, 0, 1]])
