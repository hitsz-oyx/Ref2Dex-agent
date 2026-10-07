"""Regression contracts for the non-training EPIC candidate conversion."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/audit/audit_epic_contact_pair.py'
spec = importlib.util.spec_from_file_location('epic_contact_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def poses(n):
    return np.tile(np.eye(4), (n, 1, 1))


def test_conventions_require_metadata_and_invert_w2c(tmp_path):
    c2w = poses(3)
    c2w[:, :3, :3] = Rotation.from_euler('z', [0, 20, 40], degrees=True).as_matrix()
    c2w[:, :3, 3] = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    np.testing.assert_allclose(audit.camera_to_world(np.linalg.inv(c2w), 'w2c'), c2w, atol=1e-6)
    np.testing.assert_allclose(audit.camera_to_world(c2w, 'c2w'), c2w, atol=1e-6)
    with pytest.raises(ValueError, match='unknown'):
        audit.camera_to_world(c2w, None)
    metadata = tmp_path / 'extrinsics_conv.json'
    metadata.write_text(json.dumps({'clip': 'w2c'}))
    assert audit.load_convention(metadata, 'clip') == 'w2c'
    with pytest.raises(ValueError, match='missing'):
        audit.load_convention(metadata, 'other_clip')


def test_invalid_low_confidence_endpoints_and_large_gaps_never_become_valid():
    frames = np.array([0, 2, 6, 8])
    targets = np.array([0, 1, 2, 3, 6, 7, 8], dtype=float)
    lo, hi, exact, gap = audit.grid_support(frames.astype(float), frames, targets)
    np.testing.assert_array_equal(gap, [False, False, False, True, False, False, False])
    # Endpoint 2 is low confidence. Exact 0 stays valid despite the bad next
    # observation; exact 6 stays valid despite the preceding large gap.
    source_valid = np.array([True, False, True, True])
    valid = audit.supported_mask(source_valid, lo, hi, gap)
    np.testing.assert_array_equal(valid, [True, False, False, False, True, True, True])
    np.testing.assert_array_equal(exact, [True, False, True, False, True, False, True])


def test_sparse_of_frame_ids_matched_by_value_and_motion_compared():
    frames = np.array([102, 104, 106, 108])
    of_ids = np.array([0, 2, 6, 8])
    contact = poses(4)
    contact[:, 0, 3] = [0, 1, 2, 3]
    of = poses(4)
    of[:, 0, 3] = [5, 5, 6, 6.5]
    result = audit.compare_object_poses(contact, np.ones(4, dtype=bool), frames, of_ids, of, 100)
    assert result['matched_frame_count'] == 3
    assert result['exact_pose_frame_match'] is False
    assert result['raw_center_discrepancy_m']['count'] == 3
    assert result['anchor_relative_translation_discrepancy_m']['max'] > 0
    assert result['single_scale_translation_fit']['applied'] is False
    assert result['used_for_training'] is False


def test_relative_geometry_is_world_frame_invariant():
    obj = poses(3)
    hand = np.tile(np.arange(66, dtype=float).reshape(1, 2, 11, 3) / 100, (3, 1, 1, 1))
    points = np.array([[0, 0, 0], [.5, .5, .5]])
    valid = np.ones((3, 2), dtype=bool)
    before = audit.relative_geometry(hand, obj, points, valid, valid[:, 0], np.array([0, 2, 4]))
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler('xyz', [20, 30, 40], degrees=True).as_matrix()
    world[:3, 3] = [1, 2, 3]
    obj_world = world @ obj
    hand_world = hand @ world[:3, :3].T + world[:3, 3]
    after = audit.relative_geometry(hand_world, obj_world, points, valid, valid[:, 0], np.array([0, 2, 4]))
    for side in ('right', 'left'):
        assert before[side]['nearest_surface_m']['median'] == pytest.approx(after[side]['nearest_surface_m']['median'])


def write_source(tmp_path, side='l'):
    n = 32
    frames = np.arange(n) * 2
    rot = np.tile(np.eye(3), (n, 1, 1))
    vertices = np.tile(np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]]), (n, 1, 1))
    hand = np.full((n, 21, 3), .02)
    source = tmp_path / 'clip.npz'
    np.savez(source, _clip='clip', _frame_num=frames,
             **{'mano.j3d.cam.r': hand * 2, 'mano.j3d.cam.l': hand,
                'right_valid': np.zeros(n), 'left_valid': np.ones(n),
                'object.v.cam': vertices, 'object.f': np.array([[[0, 1, 2], [0, 1, 3]]]),
                'object.rot': rot, 'object.cam_t': np.zeros((n, 3)),
                'object.diameter': .1, 'clip_verified': True})
    quality = tmp_path / 'quality.csv'
    with quality.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['clip', 'frame_num', 'high_confidence', 'hand_valid_ok', 'annotated_side'])
        writer.writeheader()
        for i, frame in enumerate(frames):
            writer.writerow(dict(clip='clip', frame_num=frame, high_confidence=int(i != 10),
                                 hand_valid_ok=1, annotated_side=side))
    return source, quality


def test_main_retains_left_hand_and_masks_low_confidence(tmp_path, monkeypatch):
    source, quality = write_source(tmp_path)
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--npz', str(source), '--quality-csv', str(quality), '--output', str(output)])
    audit.main()
    with np.load(output / 'contact_candidate.npz') as z:
        assert z['hand'].shape[1:] == (2, 11, 3)
        assert z['hand_valid'].shape == z['hand'].shape[:2]
        assert not z['hand_valid'][:, 0].any()
        assert z['hand_valid'][:, 1].any()
        assert not z['hand_valid'][:, 1].all()
        assert not z['pose_valid'].all()
        assert np.all(z['hand'][~z['hand_valid']] == 0)
        assert not z['source_hand_valid'][10].any()
    result = json.loads((output / 'audit.json').read_text())
    assert result['annotated_side'] == 'left'
    assert result['left_hand_present'] is True
    assert result['training_allowed'] is False
    assert result['status'] == 'CANDIDATE_ONLY'
    assert result['interpolated_4_plus_24_windows'] == 0


def test_main_applies_same_w2c_conversion_to_both_hands_and_object(tmp_path, monkeypatch):
    source, quality = write_source(tmp_path)
    # Enable both independently present hands, keeping the quality annotated left.
    with np.load(source, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    data['right_valid'] = np.ones(32)
    np.savez(source, **data)
    camera = poses(64)
    camera[:, 0, 3] = -1
    camera_npz = tmp_path / 'spatracker.npz'
    np.savez(camera_npz, extrinsics=camera)
    metadata = tmp_path / 'extrinsics_conv.json'
    metadata.write_text(json.dumps({'of_clip': 'w2c'}))
    output = tmp_path / 'world'
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--npz', str(source), '--quality-csv', str(quality),
        '--output', str(output), '--objectforesight-spatracker', str(camera_npz),
        '--objectforesight-start-frame', '0', '--objectforesight-clip', 'of_clip',
        '--objectforesight-conventions', str(metadata)])
    audit.main()
    with np.load(output / 'contact_candidate.npz') as z:
        np.testing.assert_allclose(z['poses'][0, 0, :3, 3], [1, 0, 0])
        np.testing.assert_allclose(z['hand'][0, 0, :, 0], 1.04)
        np.testing.assert_allclose(z['hand'][0, 1, :, 0], 1.02)
    result = json.loads((output / 'audit.json').read_text())
    assert result['camera_extrinsics']['source_convention'] == 'w2c'
    assert result['camera_extrinsics']['convention'] == 'c2w'
    assert result['training_allowed'] is False
