"""Small deterministic contracts for the EPIC scene-flow conversion probe."""
import importlib.util
import csv
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/audit/convert_epic_scene_flow.py'
spec = importlib.util.spec_from_file_location('epic_scene_flow_converter', SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def test_fit_similarity_recovers_unordered_metric_transform():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(180, 3)) * np.array([.3, .2, .02])
    rotation = Rotation.from_euler('xyz', [20, -15, 35], degrees=True).as_matrix()
    scale = 1.7
    translation = np.array([.2, -.1, .8])
    target = scale * source @ rotation + translation
    # Reordering exercises the point-cloud (rather than vertex-correspondence)
    # path used for Contact mesh to depth-mask fitting.
    target = target[rng.permutation(len(target))]
    fit = converter.fit_similarity(source, target, iterations=10)
    assert fit is not None
    assert fit['rms'] < 1e-5
    np.testing.assert_allclose(fit['scale'], scale, rtol=1e-4, atol=1e-6)


def test_camera_world_roundtrip_uses_row_points():
    w2c = np.eye(4)
    w2c[:3, :3] = Rotation.from_euler('z', 25, degrees=True).as_matrix()
    w2c[:3, 3] = [.4, -.2, 1.1]
    points = np.array([[.1, .2, .7], [-.3, .5, 1.2]])
    np.testing.assert_allclose(
        converter.camera_to_world(converter.world_to_camera(points, w2c), w2c),
        points, atol=1e-7)


def test_infer_extrinsics_convention_requires_clear_reprojection_winner():
    diagnostic = {
        'as_c2w': {'median_m': .088, 'p95_m': .27},
        'as_w2c': {'median_m': .0034, 'p95_m': .028},
    }
    result = converter.infer_extrinsics_convention(diagnostic, min_ratio=5.)
    assert result['convention'] == 'w2c'
    assert result['separation_ratio'] > 20
    with pytest.raises(ValueError, match='ambiguous'):
        converter.infer_extrinsics_convention({
            'as_c2w': {'median_m': .01, 'p95_m': .02},
            'as_w2c': {'median_m': .012, 'p95_m': .03},
        }, min_ratio=5.)


def test_clock_uses_padded_metadata_start_and_actual_rate():
    meta = dict(start_frame=1929, stop_frame=2087, pad=5, fps=60000 / 1001)
    start, fps = converter.source_clock(meta, 158)
    assert start == 1929
    assert 2 / fps != 1 / 30
    with pytest.raises(ValueError, match='disagrees'):
        converter.source_clock(meta, 158, requested_start=1927)
    with pytest.raises(ValueError, match='frame count'):
        converter.source_clock(meta, 157)


def test_lost_lk_identity_cannot_revive(monkeypatch):
    corners = np.tile([[[3., 3.]]], (16, 1, 1)).astype('float32')
    monkeypatch.setattr(converter.cv2, 'goodFeaturesToTrack', lambda *a, **k: corners)
    calls = []
    def flow(previous, image, p, *args, **kwargs):
        calls.append(len(p))
        status = np.ones((len(p), 1), dtype='uint8')
        if len(calls) == 1:
            status[0] = 0
        return p.copy(), status, None
    monkeypatch.setattr(converter.cv2, 'calcOpticalFlowPyrLK', flow)
    tracks = converter.track_object_rgb(np.zeros((3, 8, 8), dtype='uint8'),
                                        np.ones((3, 8, 8), bool), np.arange(3))
    assert calls == [16, 15]
    assert np.isnan(tracks[1:, 0]).all()
    assert np.isfinite(tracks[:, 1:]).all()


def test_contact_requires_joints_quality_and_bracketed_support(tmp_path):
    contact = tmp_path / 'fixture.npz'
    joints = np.ones((3, 21), dtype='float32')
    joints[1, converter.SEMANTIC_11[3]] = 0
    np.savez(contact, _frame_num=[10, 12, 14],
             **{'mano.j3d.cam.r': np.ones((3, 21, 3)),
                'mano.j3d.cam.l': np.ones((3, 21, 3)),
                'right_valid': np.ones(3), 'left_valid': np.zeros(3),
                'joints_valid_r': joints, 'joints_valid_l': np.zeros((3, 21)),
                'is_valid': np.ones(3), 'object.v.cam': np.zeros((3, 32, 3)),
                'object.cam_t': np.zeros((3, 3)), 'object.rot': np.tile(np.eye(3), (3, 1, 1))})
    quality = tmp_path / 'quality.csv'
    with quality.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['clip', 'frame_num', 'annotated_side',
                                              'hand_valid_ok', 'high_confidence', 'clip_verified'])
        writer.writeheader()
        for frame in [10, 12, 14]:
            writer.writerow(dict(clip='fixture', frame_num=frame, annotated_side='r',
                                  hand_valid_ok=1, high_confidence=int(frame != 14),
                                  clip_verified=True))
    out, valid, _, _, audit = converter.load_contact_hand(
        contact, np.array([-2, 0, 1, 2, 4, 6]), 10, quality_csv=quality)
    assert valid[:, 0].tolist() == [False, True, False, False, False, False]
    assert not valid[:, 1].any()
    assert not out[~valid].any()
    assert audit['outside_source_support_frames'] == 2
    with pytest.raises(ValueError, match='quality CSV'):
        converter.load_contact_hand(contact, np.array([0]), 10)


def test_test_contact_cannot_be_promoted_by_objectforesight_train(tmp_path):
    train, val = tmp_path / 'train.txt', tmp_path / 'val.txt'
    quality = tmp_path / 'epic_contact_test_frame_quality.csv'
    train.write_text('P03_13_12/objects/0+object_0\n')
    val.write_text('')
    quality.write_text('clip\n')
    result = converter.source_splits('P03_13_12', quality, train, val)
    assert result['objectforesight'] == 'train'
    assert result['epic_contact'] == 'test'
    assert not result['train_eligible']


def test_static_tracks_exclude_later_hand_occlusion():
    depths = np.ones((2, 10, 10), dtype='float32')
    intrinsics = np.tile(np.eye(3), (2, 1, 1))
    poses = np.tile(np.eye(4), (2, 1, 1))
    objects = np.zeros_like(depths, dtype=bool)
    hands = objects.copy()
    hands[1, 2, 2] = True
    _, valid, _ = converter.static_tracks(depths, intrinsics, poses, objects, hands,
                                          np.array([0, 1]), spacing=2)
    # The obscured track cannot satisfy the >=.8 selection threshold.
    assert valid.shape == (2, 8)
    assert valid.all()


def test_inventory_counts_clips_with_missing_clock_metadata(tmp_path):
    audit_spec = importlib.util.spec_from_file_location(
        'epic_video_readiness', SCRIPT.with_name('audit_epic_video_readiness.py'))
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    (tmp_path / 'shard').mkdir()
    (tmp_path / 'epic_contact').mkdir()
    with tarfile.open(tmp_path / 'shard/sample.tar', 'w') as tar:
        for name, content in [('P01_03/P01_03_3/action.mp4', b'video-placeholder'),
                              ('P03_03/P03_03_23/action.meta.json',
                               json.dumps(dict(start_frame=10, stop_frame=70, fps=60)).encode())]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    result = audit.audit(tmp_path)
    assert result['local_scene_count'] == 2
    assert result['scene_clock_metadata_count'] == 1
    assert result['scenes_missing_clock_metadata'] == ['P01_03_3']
    assert result['hand_train_qualified_windows'] == 0


def test_forward_backward_inconsistency_kills_track(monkeypatch):
    corners = np.tile([[[3., 3.]]], (16, 1, 1)).astype('float32')
    monkeypatch.setattr(converter.cv2, 'goodFeaturesToTrack', lambda *a, **k: corners)
    calls = []
    def flow(previous, image, p, *args, **kwargs):
        calls.append(len(p))
        q = p.copy()
        if len(calls) == 2:
            q[0, 0, 0] += 2
        return q, np.ones((len(p), 1), dtype='uint8'), None
    monkeypatch.setattr(converter.cv2, 'calcOpticalFlowPyrLK', flow)
    tracks = converter.track_object_rgb(np.zeros((3, 8, 8), dtype='uint8'),
                                        np.ones((3, 8, 8), bool), np.arange(3),
                                        forward_backward_threshold_px=1.)
    assert calls == [16, 16, 15, 15]
    assert np.isnan(tracks[1:, 0]).all()
