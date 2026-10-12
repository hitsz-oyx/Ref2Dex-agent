"""Shared-frame identity and provenance boundaries of trajectory-first extraction."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/audit'))
from trajectory_geometry import shared_trajectory_depth


@pytest.fixture
def geometry(tmp_path):
    ids = np.arange(0, 46, 3)
    camera = np.broadcast_to(np.eye(4), (46, 4, 4)).copy()
    intrinsic = np.eye(3)
    depth = np.broadcast_to((1+ids/100)[:, None, None], (16, 360, 640)).astype('float32')
    path = tmp_path/'depth.npz'
    np.savez_compressed(path, source_frame_ids=ids, depth_m=depth,
                        intrinsic=intrinsic, camera_to_origin=camera[ids])
    video = tmp_path/'source.mp4'
    metadata = dict(status='COMPLETED', engineering_only=True, training_allowed=False,
                    video=str(video), frames=ids.tolist(), calibration=dict(scale=.7),
                    output_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (tmp_path/'result.json').write_text(json.dumps(metadata))
    return path, intrinsic, camera, video


def test_overlapping_windows_get_identical_existing_depth_without_recalibration(geometry):
    path, intrinsic, camera, video = geometry
    a, ca, provenance = shared_trajectory_depth(path, np.arange(0, 31, 3), intrinsic, camera, video)
    b, cb, _ = shared_trajectory_depth(path, np.arange(15, 46, 3), intrinsic, camera, video)
    np.testing.assert_array_equal(a[5:], b[:6])
    assert ca == cb == dict(scale=.7)
    assert provenance['depth_context_frame_ids'] == list(range(0, 46, 3))


def test_uncovered_raw_frame_cannot_use_a_neighbor(geometry):
    path, intrinsic, camera, video = geometry
    with pytest.raises(ValueError, match='cover requested'):
        shared_trajectory_depth(path, [1, 3, 6], intrinsic, camera, video)


def test_other_recording_and_camera_are_rejected(geometry):
    path, intrinsic, camera, video = geometry
    with pytest.raises(ValueError, match='source mismatch'):
        shared_trajectory_depth(path, [3], intrinsic, camera, video.with_name('other.mp4'))
    camera[3, 0, 3] = .1
    with pytest.raises(ValueError, match='camera mismatch'):
        shared_trajectory_depth(path, [3], intrinsic, camera, video)


def test_changed_depth_cannot_keep_old_manifest(geometry):
    path, intrinsic, camera, video = geometry
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='identity drift'):
        shared_trajectory_depth(path, [3], intrinsic, camera, video)
