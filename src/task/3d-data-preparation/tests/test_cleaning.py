"""Known-answer data quality gates, immutable geometry, and heldout ownership."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pose_mesh_data.cleaning import (CleaningPolicy, clean_sequence, normalize_geometry,
    reconcile_files, window_ticks)
from pose_mesh_data.clean_data import CleanPoseMeshWindows, CleanWindow, clean_collate
from pose_mesh_data.windows import PoseMeshWindows

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/run'))
from clean_native_corpus import build


def sequence(n=80):
    hand = np.zeros((n, 2, 11, 3), np.float32)
    hand[:, 0, :, 0] = np.arange(n)[:, None]/100
    valid = np.zeros((n, 2, 11), bool)
    valid[:, 0] = True
    poses = np.broadcast_to(np.eye(4), (n, 1, 4, 4)).copy()
    poses[:, 0, 1, 3] = np.arange(n)/1000
    return dict(time=np.arange(n)/30, hand=hand, hand_valid=valid,
        object_poses=poses, object_pose_valid=np.ones((n, 1), bool),
        segment=np.zeros(n, np.int64), native_frame_id=np.arange(n))


def geometry():
    points = np.column_stack((np.linspace(0, .1, 32), np.zeros((32, 2)))).astype(np.float32)
    return dict(surface=points, query=points.copy(), normals=np.tile([2*np.pi, 0, 0], (32, 1)))


def row():
    return dict(format='canonical', source_id=8, profile='test-native11', split='train',
                target_slot=0, objects=['test'], group_id='test/train')


def test_normal_repair_preserves_query_position_and_sign_without_mutating_input():
    original = geometry()
    cleaned, report = normalize_geometry(original, CleaningPolicy())
    np.testing.assert_array_equal(cleaned['surface'], original['surface'])
    np.testing.assert_array_equal(cleaned['query'], original['query'])
    np.testing.assert_allclose(cleaned['normals'], np.tile([1, 0, 0], (32, 1)))
    np.testing.assert_allclose(original['normals'][:, 0], 2*np.pi)
    assert report['rescaled_points'] == 32
    original['normals'][2] = 0
    with pytest.raises(ValueError, match='zero_or_tiny'):
        normalize_geometry(original, CleaningPolicy())


def test_one_complete_hand_is_sufficient_and_invalid_target_interior_is_rejected():
    s = sequence()
    ticks, _ = window_ticks(s, np.ones(80, bool), 0, CleaningPolicy())
    assert ticks.tolist() == list(range(3, 56, 8))
    # Existing endpoint-only indexing would accept t=3, but t+7 is missing.
    s['object_pose_valid'][10, 0] = False
    ticks, report = window_ticks(s, np.ones(80, bool), 0, CleaningPolicy())
    assert 3 not in ticks and report['rejected']['target_future'] > 0
    s = sequence()
    s['hand_valid'][4:8, 0] = False
    ticks, report = window_ticks(s, np.ones(80, bool), 0, CleaningPolicy())
    assert 3 not in ticks and report['rejected']['hand_future'] > 0


def test_zero_nonquery_surface_points_can_be_dropped_with_fixed_query_identity():
    points = np.column_stack((np.linspace(0, 1, 512), np.zeros((512, 2)))).astype(np.float32)
    normals = np.tile([1., 0, 0], (512, 1))
    normals[400] = 0
    raw = dict(surface=points, normals=normals, query=points[:32].copy())
    cleaned, audit = normalize_geometry(raw, CleaningPolicy())
    assert len(cleaned['surface']) == 511 and audit['dropped_nonquery_points'] == 1
    np.testing.assert_array_equal(cleaned['query'], points[:32])
    np.testing.assert_array_equal(cleaned['surface'], np.delete(points, 400, axis=0))
    assert len(raw['surface']) == 512 and not raw['normals'][400].any()


def test_native_gap_marks_boundary_and_never_compresses_or_deletes_frames():
    s = sequence()
    s['native_frame_id'][40:] += 3
    cleaned, quality, audit = clean_sequence(s, row(), dict(clock_kind='fixed_fps', source_fps=30), CleaningPolicy())
    assert len(cleaned['time']) == 80 and audit['native_gaps'] == 1
    np.testing.assert_array_equal(cleaned['hand'], s['hand'])
    ticks, report = window_ticks(cleaned, quality, 0, CleaningPolicy())
    assert 19 not in ticks and report['rejected']['reset_or_clock_gap'] > 0
    assert audit['clock_verified']
    _, _, pending = clean_sequence(s, row(), dict(clock_kind='pending', source_fps=None), CleaningPolicy())
    assert pending['clock_verified'] is False


def test_timestamp_qa_repeats_and_alignment_masks_are_not_interpolated():
    s = sequence()
    grid = np.rint(np.arange(80)*1e9/30).astype(np.int64)+10**12
    native = grid.copy()
    native[20] = native[19]
    native[60] += 17_000_000
    qa = np.ones(80, bool)
    qa[35] = False
    s.update(native_timestamp_ns=native, grid_timestamp_ns=grid,
             alignment_residual_ns=native-grid, native_qa_pass=qa)
    cleaned, quality, audit = clean_sequence(s, row(), dict(clock_kind='timestamps'), CleaningPolicy())
    assert not quality[[20, 35, 60]].any()
    assert not cleaned['hand_valid'][[20, 35, 60]].any()
    assert not cleaned['object_pose_valid'][[20, 35, 60]].any()
    assert audit['repeated_or_backward_timestamps'] == 1
    assert s['hand_valid'][35, 0].all()  # Original arrays are preserved.


def test_heldout_test_files_have_owners_and_cannot_overlap_train(tmp_path):
    (tmp_path/'sequences').mkdir()
    for name in ('train', 'test'):
        np.savez(tmp_path/'sequences'/(name+'.npz'), x=[1])
    a = dict(sequences=[dict(path='sequences/train.npz')])
    b = dict(sequences=[dict(path='sequences/test.npz')])
    result = reconcile_files(tmp_path, a, b)
    assert result['heldout_test'] == 1 and not result['unclaimed'] and not result['missing']
    with pytest.raises(ValueError, match='overlaps'):
        reconcile_files(tmp_path, a, a)


def test_frozen_pipeline_reads_both_splits_and_blocks_legacy_bypass(tmp_path, monkeypatch):
    # Keep all fixture artifacts in the repository's pytest basetemp.
    import clean_native_corpus
    monkeypatch.setattr(clean_native_corpus, 'REPO', tmp_path)
    monkeypatch.setattr(clean_native_corpus.subprocess, 'check_output', lambda *a, **k: 'test-commit\n')
    root = tmp_path/'native'
    root.mkdir()
    (root/'sequences').mkdir()
    np.savez(root/'object.npz', **geometry())
    rows = []
    for split in ('train', 'dev'):
        np.savez(root/'sequences'/(split+'.npz'), **sequence())
        rows.append({**row(), 'path':'sequences/'+split+'.npz', 'split':split, 'group_id':'test/'+split})
    source = dict(schema='ref2dex.pose-mesh.v1', units='m', fps=30, training_allowed=False,
        sequences=rows, objects=[dict(key='test', path='object.npz')], source_manifests=[str(root)])
    (root/'manifest.json').write_text(json.dumps(source))
    output = tmp_path/'outputs/3d-data-preparation/clean-test'
    output.parent.mkdir(parents=True)
    config = tmp_path/'policy.json'
    config.write_text(json.dumps(dict(policy=CleaningPolicy().to_dict(), sources={
        '8':dict(clock_kind='fixed_fps', source_fps=30)})))
    result = build(root, output, config, workers=1)
    assert result['windows']['verified_train'] == 7 and result['windows']['verified_dev'] == 7
    assert json.loads((root/'manifest.json').read_text()) == source
    with pytest.raises(ValueError, match='CleanPoseMeshWindows'):
        PoseMeshWindows(output, 'train', engineering=True)
    ds = CleanPoseMeshWindows(output, 'train', engineering=True)
    sample = ds[0]
    batch = clean_collate([sample])
    assert batch['training_allowed'] is False and batch['provenance'][0].clock_verified
    assert batch['model_inputs']['source_id'].tolist() == [8]
    assert len(batch['provenance'][0].observed_future) == 24
    assert all(batch['provenance'][0].observed_future)
    forged = dict(sample.model_inputs)
    forged['anchor_world_to_sample'] = forged['anchor_world_to_sample'].copy()
    forged['anchor_world_to_sample'][0,3] += 1
    with pytest.raises(ValueError, match='identity'):
        clean_collate([CleanWindow(forged,sample.provenance)])
    np.testing.assert_allclose(sample.model_inputs['scene_target'][0, -1, :, 1], .024, atol=1e-6)
    with pytest.raises(ValueError, match='truncation'):
        clean_collate([sample], scene_cap=16)
    # Drift is rejected even after a sequence has entered the reader cache.
    path = root/'sequences/train.npz'
    with path.open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        ds[0]
    with (output/'windows.npz').open('ab') as f:
        f.write(b'changed-index')
    with pytest.raises(ValueError, match='window index changed'):
        ds[0]
    # Final budget failure must not publish a readable COMPLETED manifest.
    failed = output.parent/'clean-test-final-budget-failure'
    monkeypatch.setattr(clean_native_corpus, 'footprint', lambda root:
        3_000_000_000 if (root/'cleaning-summary.json').exists() else 0)
    with pytest.raises(RuntimeError, match='artifact budget'):
        build(root, failed, config, workers=1)
    assert not (failed/'manifest.json').exists()
    assert json.loads((failed/'progress.json').read_text())['status'] == 'FAILED'
