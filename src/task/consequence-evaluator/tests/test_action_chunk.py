import json

import numpy as np
import pytest
import torch

from consequence_evaluator.action_chunk import (
    K, NativeActionChunkProposal, HistoryStandardizer, chunk_metrics,
    chunk_windows, load_action_chunk_batches, validate_action_episode)


def test_chunk_windows_never_cross_episode_and_preserves_tick():
    history = np.arange(9 * 3, dtype='float32').reshape(9, 3)
    action = np.arange(8 * 18, dtype='float32').reshape(8, 18) / 1000
    h, a = chunk_windows(history, action, chunk=4, stride=2)
    assert h.shape == (3, 3)
    assert a.shape == (3, 4, 18)
    np.testing.assert_array_equal(h[1], history[2])
    np.testing.assert_array_equal(a[1], action[2:6])


def test_episode_contract_rejects_padding_and_out_of_range():
    with pytest.raises(ValueError):
        validate_action_episode(np.zeros((25, 3)), np.zeros((25, 18)))
    with pytest.raises(ValueError):
        validate_action_episode(np.zeros((25, 3)), np.full((24, 18), 1.1))


def test_model_is_one_shot_bounded_native_chunk():
    model = NativeActionChunkProposal(7, width=32, layers=1)
    output = model(torch.zeros(3, 7))
    assert output.shape == (3, K, 18)
    assert torch.isfinite(output).all()
    assert float(output.abs().max()) <= 1 + 1e-6
    with pytest.raises(ValueError):
        model(torch.zeros(3, 6))


def test_standardizer_and_metrics_are_train_only_utilities():
    x = np.arange(12, dtype='float32').reshape(4, 3)
    standardizer = HistoryStandardizer.fit(x)
    np.testing.assert_allclose(standardizer.transform(x).mean(0), 0, atol=1e-6)
    target = torch.zeros(2, K, 18)
    prediction = torch.ones_like(target) * .1
    metrics = chunk_metrics(prediction, target, torch.zeros(K, 18))
    assert metrics['mse'] == pytest.approx(.01)
    assert metrics['mse_ratio_to_mean'] is None
    native = HistoryStandardizer.from_running_stats(np.zeros(3), np.ones(3))
    assert float(np.abs(native.transform(np.full((1, 3), 100))).max()) == pytest.approx(5.)


def test_loader_selects_clean_success_only(tmp_path):
    root = tmp_path / 'run'; root.mkdir()
    t = 30
    history = np.zeros((t + 1, 1442), dtype='float32')
    action = np.zeros((t, 18), dtype='float32')
    residual = np.zeros((t, K, 18), dtype='float32')
    arrays = dict(history=history, action=action, residual_plan=residual,
                  plan_known=np.ones(t, dtype=bool), hand_keypoints=np.zeros((t+1,11,3), dtype='float32'),
                  object_pose=np.tile(np.eye(4, dtype='float32'), (t+1,1,1)),
                  timestamps=np.arange(t+1, dtype='float32') / 30,
                  phase=np.full(t, 'hold'), progress=np.full(t+1, np.nan, dtype='float32'),
                  progress_mask=np.zeros(t+1, dtype=bool))
    path = root / 'episode.npz'; np.savez_compressed(path, **arrays)
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(schema='ref2dex.consequence-evaluator.episodes.v2', status='COMPLETED',
                    rollout_kind='continuous', action_semantics='decision_known_requested_residual_plan',
                    fps=30, units='m', training_allowed=True,
                    history_contract={'source': 'raw native policy observation; per-expert RMS remains inside player',
                                      'shape': [1442]},
                    episodes=[dict(episode='e', split='train', quality='expert_success',
                                   assigned_phase='clean', perturbation_tick=-1,
                                   path='episode.npz', sha256=digest)])
    (root / 'manifest.json').write_text(json.dumps(manifest))
    batches, _ = load_action_chunk_batches(root, stride=4, splits=('train',))
    assert batches['train'].action.shape == (2, K, 18)


def test_loader_clean_only_skips_unlabeled_intervention(tmp_path):
    root = tmp_path / 'run'; root.mkdir()
    t = 24
    history = np.zeros((t + 1, 1442), dtype='float32')
    action = np.zeros((t, 18), dtype='float32')
    residual = np.zeros((t, K, 18), dtype='float32')
    common = dict(history=history, action=action, residual_plan=residual,
                  plan_known=np.ones(t, dtype=bool), hand_keypoints=np.zeros((t+1,11,3), dtype='float32'),
                  object_pose=np.tile(np.eye(4, dtype='float32'), (t+1,1,1)),
                  timestamps=np.arange(t+1, dtype='float32') / 30,
                  phase=np.full(t, 'hold'), progress=np.full(t+1, np.nan, dtype='float32'),
                  progress_mask=np.zeros(t+1, dtype=bool))
    path = root / 'episode.npz'; np.savez_compressed(path, **common)
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(schema='ref2dex.consequence-evaluator.episodes.v2', status='COMPLETED',
                    rollout_kind='continuous', action_semantics='decision_known_requested_residual_plan',
                    fps=30, units='m', training_allowed=True,
                    history_contract={'source': 'raw native policy observation; per-expert RMS remains inside player',
                                      'shape': [1442]},
                    episodes=[dict(episode='e', split='train', quality='unlabeled',
                                   assigned_phase='approach', perturbation_tick=1,
                                   path='episode.npz', sha256=digest)])
    (root / 'manifest.json').write_text(json.dumps(manifest))
    batches, _ = load_action_chunk_batches(root, qualities=('unlabeled',), clean_only=True,
                                           stride=4, splits=('train',))
    assert batches['train'].action.shape == (0, K, 18)
    manifest['episodes'][0].update(assigned_phase='clean', perturbation_tick=-1.5)
    (root / 'manifest.json').write_text(json.dumps(manifest))
    batches, _ = load_action_chunk_batches(root, qualities=('unlabeled',), clean_only=True,
                                           stride=4, splits=('train',))
    assert batches['train'].action.shape == (0, K, 18)


def test_loader_clean_only_validates_unlabeled_zero_plan(tmp_path):
    root = tmp_path / 'run'; root.mkdir()
    t = 24
    arrays = dict(history=np.zeros((t + 1, 1442), dtype='float32'),
                  action=np.zeros((t, 18), dtype='float32'),
                  residual_plan=np.zeros((t, K, 18), dtype='float32'),
                  plan_known=np.ones(t, dtype=bool),
                  hand_keypoints=np.zeros((t+1,11,3), dtype='float32'),
                  object_pose=np.tile(np.eye(4, dtype='float32'), (t+1,1,1)),
                  timestamps=np.arange(t+1, dtype='float32') / 30,
                  phase=np.full(t, 'hold'), progress=np.full(t+1, np.nan, dtype='float32'),
                  progress_mask=np.zeros(t+1, dtype=bool))
    path = root / 'episode.npz'; np.savez_compressed(path, **arrays)
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(schema='ref2dex.consequence-evaluator.episodes.v2', status='COMPLETED',
                    rollout_kind='continuous', action_semantics='decision_known_requested_residual_plan',
                    fps=30, units='m', training_allowed=True,
                    history_contract={'source': 'raw native policy observation; per-expert RMS remains inside player',
                                      'shape': [1442]},
                    episodes=[dict(episode='e', split='train', quality='unlabeled',
                                   assigned_phase='clean', perturbation_tick=-1,
                                   path='episode.npz', sha256=digest)])
    (root / 'manifest.json').write_text(json.dumps(manifest))
    batches, _ = load_action_chunk_batches(root, qualities=('unlabeled',), clean_only=True,
                                            stride=4, splits=('train',))
    assert batches['train'].action.shape == (1, K, 18)


def test_default_expert_loader_keeps_unknown_plan_compatibility(tmp_path):
    root = tmp_path / 'run'; root.mkdir()
    t = 24
    arrays = dict(history=np.zeros((t + 1, 1442), dtype='float32'),
                  action=np.zeros((t, 18), dtype='float32'),
                  residual_plan=np.zeros((t, K, 18), dtype='float32'),
                  plan_known=np.zeros(t, dtype=bool),
                  hand_keypoints=np.zeros((t+1,11,3), dtype='float32'),
                  object_pose=np.tile(np.eye(4, dtype='float32'), (t+1,1,1)),
                  timestamps=np.arange(t+1, dtype='float32') / 30,
                  phase=np.full(t, 'hold'), progress=np.full(t+1, np.nan, dtype='float32'),
                  progress_mask=np.zeros(t+1, dtype=bool))
    path = root / 'episode.npz'; np.savez_compressed(path, **arrays)
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(schema='ref2dex.consequence-evaluator.episodes.v2', status='COMPLETED',
                    rollout_kind='continuous', action_semantics='decision_known_requested_residual_plan',
                    fps=30, units='m', training_allowed=True,
                    history_contract={'source': 'raw native policy observation; per-expert RMS remains inside player',
                                      'shape': [1442]},
                    episodes=[dict(episode='e', split='train', quality='expert_success',
                                   assigned_phase='clean', perturbation_tick=-1,
                                   path='episode.npz', sha256=digest)])
    (root / 'manifest.json').write_text(json.dumps(manifest))
    batches, _ = load_action_chunk_batches(root, stride=4, splits=('train',))
    assert batches['train'].action.shape == (1, K, 18)
    manifest['episodes'][0]['perturbation_tick'] = -1.5
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unperturbed clean'):
        load_action_chunk_batches(root, stride=4, splits=('train',))
    manifest['episodes'][0]['perturbation_tick'] = -1
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unknown residual-plan'):
        load_action_chunk_batches(root, clean_only=True, stride=4, splits=('train',))
