"""Scientific contract regressions; tiny CPU models are engineering checks only."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / 'src'))
sys.path.insert(0, str(TASK / 'tools/run'))
from consequence_evaluator.data import K, Windows, object_effect, sha
from consequence_evaluator.model import Evaluator, matched_loss, progress_loss
from consequence_evaluator.contracts import (EPISODE_SCHEMA, ACTION_SEMANTICS, FUTURE_DIM,
                                             HISTORY_MATCH_MAX_RELATIVE_RMS)
from consequence_evaluator.supervision import CONTACT_SEMANTICS, RULE
from prepare_windows import prepare


def fixture(root, mutation=None):
    root.mkdir()
    records = []
    pairs = []
    for split in ('train', 'val', 'test'):
        for quality in ('expert_success', 'failure'):
            episode = split + '_' + quality
            steps = 25
            pose = np.broadcast_to(np.eye(4), (steps + 1, 4, 4)).copy()
            pose[:, 2, 3] = np.arange(steps + 1) / 1000
            packet = dict(history=np.ones((steps + 1, 6), dtype='float32'),
                          residual_plan=np.zeros((steps,K,18),dtype='float32'),plan_known=np.ones(steps,dtype=bool),
                          hand_keypoints=np.zeros((steps+1,11,3),dtype='float32'),
                          action=np.zeros((steps, 18), dtype='float32'), object_pose=pose,
                          timestamps=np.arange(steps + 1) / 30,
                          phase=np.asarray(['lift'] * steps),
                          progress=np.arange(steps + 1) / steps,
                          progress_mask=np.full(steps + 1, quality == 'expert_success', dtype=bool))
            if mutation:
                mutation(packet, split, quality)
            path = root / (episode + '.npz')
            np.savez_compressed(path, **packet)
            records.append(dict(episode=episode, split=split, split_group=episode,
                                quality=quality, task='airplane',expert='synthetic',motion='fixture', path=path.name, sha256=sha(path)))
        pairs.append(dict(chosen=dict(episode=split + '_expert_success', tick=0),
                          rejected=dict(episode=split + '_failure', tick=0),
                          annotation='synthetic-local-event-engineering-only'))
    manifest = dict(schema=EPISODE_SCHEMA,action_semantics=ACTION_SEMANTICS, rollout_kind='continuous',
                    status='COMPLETED',
                    training_allowed=True, pair_coverage_required=False, fps=30, units='m', history_contract='synthetic current observation only',
                    contact_semantics=CONTACT_SEMANTICS, label_rule=RULE,
                    episodes=records)
    (root / 'manifest.json').write_text(json.dumps(manifest))
    labels = root / 'preferences.json'
    labels.write_text(json.dumps(dict(scope='local_window', label_provenance=dict(
        rule=RULE, contact_semantics=CONTACT_SEMANTICS,
        rule_sha256=sha(TASK/'src/consequence_evaluator/supervision.py'),
        contracts_sha256=sha(TASK/'src/consequence_evaluator/contracts.py'),
        history_match_relative_rms=HISTORY_MATCH_MAX_RELATIVE_RMS,
        minimum_pair_coverage={'train': 8, 'val': 4, 'test': 4}), pairs=pairs)))
    return labels


def change_arrays(root, change):
    path = root / 'windows.npz'
    with np.load(path, allow_pickle=False) as packet:
        arrays = {name: packet[name].copy() for name in packet.files}
    change(arrays)
    np.savez_compressed(path, **arrays)
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['windows_sha256'] = sha(path)
    (root / 'manifest.json').write_text(json.dumps(manifest))


def test_prepost_alignment_absolute_progress_and_input_whitelist(tmp_path):
    source, out = tmp_path / 'source', tmp_path / 'out'
    labels = fixture(source)
    report = prepare(source, labels, out)
    assert report['windows'] == 12 and report['pairs'] == dict(train=1, val=1, test=1)
    data = Windows(out)
    inputs, supervision = data.batch([0], 'cpu')
    assert set(inputs) == {'history', 'action', 'future'}
    assert supervision['progress'][0, 0] == pytest.approx(1 / 25)
    assert supervision['progress'][0, -1] == pytest.approx(24 / 25)
    assert data.arrays['effect'][0, 0, 2, 3] == pytest.approx(.001)
    assert data.arrays['effect'][0, -1, 2, 3] == pytest.approx(.024)


@pytest.mark.parametrize('kind', ['episode_split', 'pair_phase', 'failure_progress', 'label_input'])
def test_rejects_leakage_and_invented_supervision(tmp_path, kind):
    source, out = tmp_path / 'source', tmp_path / 'out'
    labels = fixture(source)
    prepare(source, labels, out)
    def mutate(a):
        if kind == 'episode_split':
            a['split'][1] = 'val'
        elif kind == 'pair_phase':
            a['phase'][2] = 'hold'
        elif kind == 'failure_progress':
            a['progress_mask'][2, 0] = True
        else:
            a['success'] = np.ones(len(a['history']))
    change_arrays(out, mutate)
    with pytest.raises(ValueError):
        Windows(out)


def test_clock_gap_rejected_before_window_generation(tmp_path):
    source = tmp_path / 'source'
    def gap(packet, split, quality):
        packet['timestamps'][12:] += .1
    labels = fixture(source, gap)
    with pytest.raises(ValueError, match='clock'):
        prepare(source, labels, tmp_path / 'out')


def test_changed_data_rejected_even_when_shapes_remain_valid(tmp_path):
    source, out = tmp_path / 'source', tmp_path / 'out'
    labels = fixture(source)
    prepare(source, labels, out)
    with (out / 'windows.npz').open('ab') as stream:
        stream.write(b'input drift')
    with pytest.raises(ValueError, match='identity'):
        Windows(out)


def test_reverse_preference_direction_is_rejected(tmp_path):
    source = tmp_path / 'source'
    labels = fixture(source)
    document = json.loads(labels.read_text())
    original = document['pairs'][0]
    document['pairs'].append(dict(
        chosen=original['rejected'], rejected=original['chosen'],
        annotation='synthetic contradictory direction'))
    labels.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='contradictory'):
        prepare(source, labels, tmp_path / 'out')


def test_fractional_preference_tick_is_rejected_without_coercion(tmp_path):
    source = tmp_path / 'source'
    labels = fixture(source)
    document = json.loads(labels.read_text())
    document['pairs'][0]['chosen']['tick'] = 0.5
    labels.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='tick'):
        prepare(source, labels, tmp_path / 'out')


def test_normalization_uses_training_inputs_only(tmp_path):
    source, out = tmp_path / 'source', tmp_path / 'out'
    labels = fixture(source)
    prepare(source, labels, out)
    original = Windows(out).normalization()
    def change(a):
        a['history'][a['split'] != 'train'] = 9000
        a['action'][a['split'] != 'train'] = .19
    change_arrays(out, change)
    modified = Windows(out).normalization()
    assert all(torch.equal(old, new) for key in original
               for old, new in zip(original[key], modified[key]))


def test_seed_group_cannot_cross_splits(tmp_path):
    source = tmp_path / 'source'
    labels = fixture(source)
    manifest = json.loads((source / 'manifest.json').read_text())
    for item in manifest['episodes']:
        item['split_group'] = 'same_collection_seed'
    (source / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='group crosses'):
        prepare(source, labels, tmp_path / 'out')


def test_unfinished_rollout_or_window_export_cannot_feed_training(tmp_path):
    source, out = tmp_path/'source', tmp_path/'out'
    labels = fixture(source)
    path=source/'manifest.json'
    manifest=json.loads(path.read_text())
    manifest['status']='RUNNING'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='provenance'):
        prepare(source, labels, out)
    manifest['status']='COMPLETED'
    path.write_text(json.dumps(manifest))
    prepare(source, labels, out)
    path=out/'manifest.json'
    manifest=json.loads(path.read_text())
    manifest['status']='PREPARED'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='completed'):
        Windows(out)


def test_effect_invariant_to_stationary_world_coordinate_change():
    poses = np.broadcast_to(np.eye(4), (K + 1, 4, 4)).copy()
    poses[:, 0, 3] = np.linspace(0, .1, K + 1)
    world = np.eye(4)
    world[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    world[:3, 3] = [1, 2, 3]
    assert np.allclose(object_effect(poses), object_effect(world @ poses), atol=1e-6)


def test_baseline_future_invariance_identical_capacity_and_initialization():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    first = Evaluator(6, width=8, layers=1)
    second = copy.deepcopy(first)
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    h, action, future = torch.randn(2, 6), torch.randn(2, K, 18), torch.randn(2, K, FUTURE_DIM)
    zero = first(h, action, future, use_future=False)
    changed = first(h, action, future + 100, use_future=False)
    assert torch.equal(zero['score'], changed['score'])
    oracle = second(h, action, future)
    oracle_changed = second(h, action, future + 1)
    assert not torch.allclose(oracle['score'], oracle_changed['score'])


def test_failed_progress_has_zero_gradient_and_endpoint_bins_work():
    logits = torch.randn(2, K, 10, requires_grad=True)
    target = torch.full((2, K), float('nan'))
    mask = torch.zeros(2, K, dtype=torch.bool)
    target[0] = torch.linspace(0, 1, K)
    mask[0] = True
    loss = progress_loss(logits, target, mask)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad[0].abs().sum() > 0
    assert logits.grad[1].abs().sum() == 0


def test_matched_cpu_optimizer_and_checkpoint_roundtrip(tmp_path):
    """Two tiny updates verify wiring; no learned headroom assertion."""
    source, out = tmp_path / 'source', tmp_path / 'out'
    labels = fixture(source)
    prepare(source, labels, out)
    data = Windows(out)
    pairs = data.arrays['pairs'][data.pair_ids['train']]
    left, ll = data.batch(pairs[:, 0], 'cpu')
    right, rl = data.batch(pairs[:, 1], 'cpu')
    initial = Evaluator(6, width=8, layers=1)
    for use_future in (False, True):
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(2):
            optimizer.zero_grad()
            loss, _ = matched_loss(model(**left, use_future=use_future), model(**right, use_future=use_future), ll, rl)
            loss.backward()
            assert torch.isfinite(loss)
            optimizer.step()
        path = tmp_path / ('oracle.pt' if use_future else 'baseline.pt')
        torch.save(model.state_dict(), path)
        restored = Evaluator(6, width=8, layers=1)
        restored.load_state_dict(torch.load(path, weights_only=True))
        assert torch.equal(model(**left, use_future=use_future)['score'], restored(**left, use_future=use_future)['score'])
