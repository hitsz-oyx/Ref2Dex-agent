"""Semantic tests: masking, correspondence and future/observation boundaries."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK/'src'))
from goal_point_wam.point_supervision import masked_point_loss, PointTrajectoryDecoder
from goal_point_wam.video_data import video_collate, VideoPointWindows


def test_nan_invalid_labels_and_duplicated_points_do_not_change_loss():
    pred = torch.tensor([[[[.01, 0, 0], [.02, 0, 0]]]], requires_grad=True)
    target = torch.zeros_like(pred)
    current = torch.zeros(1, 2, 3)
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    baseline = masked_point_loss(pred, target, valid, current)
    duplicate = masked_point_loss(pred.repeat_interleave(3, 2), target.repeat_interleave(3, 2),
                                  valid.repeat_interleave(3, 2), current.repeat_interleave(3, 1))
    torch.testing.assert_close(baseline, duplicate)
    invalid_pred = torch.cat((pred, torch.full((1, 1, 1, 3), float('nan'))), 2)
    invalid_target = torch.cat((target, torch.full((1, 1, 1, 3), float('nan'))), 2)
    masked = masked_point_loss(invalid_pred, invalid_target, torch.tensor([[[True, True, False]]]),
                               torch.zeros(1, 3, 3))
    torch.testing.assert_close(baseline, masked)
    masked.backward()
    assert torch.isfinite(pred.grad).all()


def test_query_permutation_and_chunking_preserve_correspondence():
    torch.manual_seed(7)
    decoder = PointTrajectoryDecoder(chunk_size=2).eval()
    torch.nn.init.normal_(decoder.head[-1].weight, std=.02)
    xyz, features = torch.randn(1, 5, 3)*.1, torch.randn(1, 5, 128)
    valid = torch.ones(1, 5, dtype=torch.bool)
    scene, action = torch.randn(1, 3, 128), torch.randn(1, 4, 128)
    sm, am = torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool)
    pred = decoder(xyz, features, valid, scene, sm, action, am)
    order = torch.tensor([4, 1, 3, 0, 2])
    permuted = decoder(xyz[:, order], features[:, order], valid[:, order], scene, sm, action, am)
    torch.testing.assert_close(pred[:, :, order], permuted, atol=1e-6, rtol=1e-5)
    decoder.chunk_size = 5
    torch.testing.assert_close(pred, decoder(xyz, features, valid, scene, sm, action, am), atol=1e-6, rtol=1e-5)


def sample(n):
    h = np.zeros((28, 2, 11, 3), 'float32')
    h[..., 0] = np.arange(28)[:, None, None]*.001
    return dict(query_xyz=np.zeros((n, 3), 'float32'), query_valid=np.ones(n, bool),
                point_target=np.zeros((24, n, 3), 'float32'), point_target_valid=np.ones((24, n), bool),
                track_ids=np.arange(n), hand=h, hand_valid=np.ones((28, 2, 11), bool),
                start_object_points=np.zeros((32, 3), 'float32'), goal_object_points=np.zeros((32, 3), 'float32'),
                object_goal_valid=np.zeros(32, bool))


def write_corpus(root, arrays):
    import hashlib
    root.mkdir()
    np.savez(root/'clip.npz', **arrays)
    meta = dict(schema='goal-point-wam.video-points.v1', status='COMPLETED', fps=30, history=4, horizon=24,
                causal_observation=True, training_allowed=False,
                records=[dict(path='clip.npz', sha256=hashlib.sha256((root/'clip.npz').read_bytes()).hexdigest(),
                              session='sessionA', split='train', source_split='official_train')])
    (root/'manifest.json').write_text(json.dumps(meta))
    return VideoPointWindows(root, 'train', engineering=True)


def test_future_labels_do_not_enter_current_features(tmp_path):
    arrays = sample(5)
    before = write_corpus(tmp_path/'before', arrays)[0]
    arrays['hand'][4:] += .3
    arrays['point_target'] += .4
    after = write_corpus(tmp_path/'after', arrays)[0]
    for key in ('query_xyz', 'features', 'hand_features', 'start_hand', 'start_hand_history', 'start_hand_valid'):
        np.testing.assert_array_equal(before[key], after[key])
    assert not np.array_equal(before['action'], after['action'])
    assert not np.array_equal(before['trajectory_target'], after['trajectory_target'])
    batch = video_collate([before, write_corpus(tmp_path/'small', sample(3))[0]])
    assert batch['query_xyz'].shape == (2, 5, 3)
    assert not batch['query_valid'][1, 3:].any()
    assert batch['point_valid'][:, 5:].all()  # hands after padded query extent
    assert not batch['object_valid'].any()


def test_reject_engineering_and_noncausal_scientific_data(tmp_path):
    dataset = write_corpus(tmp_path/'corpus', sample(4))
    with pytest.raises(ValueError, match='engineering corpus'):
        VideoPointWindows(dataset.root, 'train')
    meta = dataset.meta
    meta['causal_observation'] = False
    (dataset.root/'manifest.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='current/past'):
        VideoPointWindows(dataset.root, 'train', engineering=True)
