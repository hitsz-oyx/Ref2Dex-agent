"""Tiny CPU checks of held-out bookkeeping, not evidence of oracle headroom."""
import copy
import json
import time

import numpy as np
import pytest
import torch

from test_consequence_evaluator import TASK, fixture
from prepare_windows import prepare
from consequence_evaluator.data import Windows, sha
from consequence_evaluator.evaluation import future_donors, predict, summarize
from consequence_evaluator.model import Evaluator
from evaluate_matched import freeze_protocol, load_frozen


def prepared(tmp_path):
    source, out = tmp_path / 'source', tmp_path / 'data'
    def different_future(packet, split, quality):
        rate = .002 if quality == 'expert_success' else -.003
        packet['object_pose'][:, 0, 3] = np.arange(len(packet['object_pose'])) * rate
    labels = fixture(source, different_future)
    # Two overlapping windows per episode pair: group metrics must not imply
    # four independent episodes or duplicate frame counts for progress.
    document = json.loads(labels.read_text())
    for pair in copy.deepcopy(document['pairs']):
        pair['chosen']['tick'] = pair['rejected']['tick'] = 1
        document['pairs'].append(pair)
    labels.write_text(json.dumps(document))
    prepare(source, labels, out)
    return Windows(out), out


def test_donors_are_deterministic_cross_episode_test_only_and_matched(tmp_path):
    data, _ = prepared(tmp_path)
    ids = np.unique(data.arrays['pairs'][data.pair_ids['test']])
    donors = future_donors(data, ids, 23)
    assert np.array_equal(donors, future_donors(data, ids, 23))
    a = data.arrays
    assert (a['episode'][ids] != a['episode'][donors]).all()
    assert (a['split'][donors] == 'test').all()
    assert (a['task'][ids] == a['task'][donors]).all()
    assert (a['phase'][ids] == a['phase'][donors]).all()
    with pytest.raises(ValueError, match='test split'):
        future_donors(data, np.array([0]), 23)


def test_donor_control_changes_only_future_and_restores_model_mode(tmp_path):
    data, _ = prepared(tmp_path)
    ids = np.unique(data.arrays['pairs'][data.pair_ids['test']])
    donors = future_donors(data, ids, 7)
    torch.set_num_threads(1)
    model = Evaluator(6, width=8, layers=1)
    original = {key: value.copy() for key, value in data.arrays.items()}
    statistics = data.normalization()
    baseline = predict(model, data, ids, statistics, 'cpu', False, batch=2)
    reassigned = predict(model, data, ids, statistics, 'cpu', False, batch=2, donors=donors)
    assert model.training
    assert all(np.array_equal(baseline[key], reassigned[key]) for key in baseline)
    oracle = predict(model, data, ids, statistics, 'cpu', True, batch=2)
    control = predict(model, data, ids, statistics, 'cpu', True, batch=2, donors=donors)
    assert not np.allclose(oracle['score'], control['score'])
    assert all(np.array_equal(old, data.arrays[key], equal_nan=True) if old.dtype.kind == 'f'
               else np.array_equal(old, data.arrays[key]) for key, old in original.items())
    model.eval()
    with pytest.raises(TimeoutError):
        predict(model, data, ids, statistics, 'cpu', True, deadline=time.monotonic() - 1)
    assert not model.training


def test_paired_metrics_masks_groups_and_ties_are_recomputed(tmp_path):
    data, _ = prepared(tmp_path)
    pairs = data.arrays['pairs'][data.pair_ids['test']]
    ids = np.unique(pairs)
    scores = dict(baseline=np.zeros(4), oracle=np.array([2., 2., 1., 1.]),
                  oracle_reassigned_future=np.array([1., 1., 2., 2.]))
    predictions = {name: dict(score=value, progress=data.arrays['progress'][ids].copy())
                   for name, value in scores.items()}
    # Invalid failure labels must not contribute even if the prediction is poor.
    for packet in predictions.values():
        packet['progress'][~data.arrays['progress_mask'][ids]] = .9
    report = summarize(data, ids, predictions)
    assert report['metrics']['baseline']['ties'] == 2
    assert report['metrics']['baseline']['strict_accuracy'] == 0
    assert report['metrics']['oracle']['strict_accuracy'] == 1
    assert report['metrics']['oracle']['episode_pair_groups'] == 1
    assert report['metrics']['oracle_minus_baseline']['oracle_only_correct'] == 2
    assert report['aligned_minus_reassigned']['accuracy_gain'] == 1
    assert report['progress_on_reliable_frames']['oracle'] == dict(labeled_frames=48, unique_windows=2, mae=0.)
    assert len(report['by_task_phase']) == 1
    assert report['by_quality'][0]['chosen'] == 'expert_success'
    with pytest.raises(ValueError, match='exactly'):
        summarize(data, ids[::-1], predictions)


def fake_fit(fit, data, root):
    fit.mkdir()
    statistics = data.normalization()
    model = Evaluator(6, width=8, layers=1)
    torch.save(dict(model=model.state_dict(), statistics=statistics), fit / 'initial.pt')
    manifest = dict(status='COMPLETED', task='consequence-evaluator', width=8, layers=1,
                    updates=10, pairs_per_update=2, windows_sha256=data.manifest['windows_sha256'],
                    window_manifest_sha256=sha(root / 'manifest.json'),
                    initial_weights_sha256=sha(fit / 'initial.pt'), training_pair_draw_sha256='engineering-only',
                    parameters=sum(v.numel() for v in model.parameters()), sources={})
    (fit / 'run_manifest.json').write_text(json.dumps(manifest))
    # Equal later validation accuracy must retain the first best checkpoint.
    records = [dict(step=step, baseline=dict(strict_accuracy=.75), oracle=dict(strict_accuracy=1.))
               for step in (5, 10)]
    (fit / 'validation.jsonl').write_text('\n'.join(json.dumps(r) for r in records) + '\n')
    for arm in ('baseline', 'oracle'):
        torch.save(dict(arm=arm, step=5, manifest=manifest, statistics=statistics,
                        model=model.state_dict()), fit / (arm + '-best.pt'))


@pytest.mark.parametrize('problem', ['unfinished', 'statistics', 'selection', 'initialization'])
def test_frozen_loader_rejects_unfinished_drift_or_test_based_selection(tmp_path, problem):
    data, root = prepared(tmp_path)
    fit = tmp_path / 'fit'
    fake_fit(fit, data, root)
    models, statistics, selections = load_frozen(fit, data, root)
    assert set(models) == set(statistics) == set(selections) == {'baseline', 'oracle'}
    assert selections['oracle']['step'] == 5
    if problem == 'unfinished':
        path = fit / 'run_manifest.json'
        m = json.loads(path.read_text())
        m['status'] = 'RUNNING'
        path.write_text(json.dumps(m))
    elif problem == 'initialization':
        with (fit / 'initial.pt').open('ab') as stream:
            stream.write(b'drift')
    else:
        path = fit / 'oracle-best.pt'
        payload = torch.load(path, weights_only=True)
        if problem == 'statistics':
            payload['statistics']['history'][0].add_(1)
        else:
            payload['step'] = 10
        torch.save(payload, path)
    with pytest.raises(ValueError):
        load_frozen(fit, data, root)


def test_frozen_test_protocol_disallows_new_weights_or_control_seed(tmp_path):
    protocol = dict(checkpoints={'oracle': 'fixed-sha'}, seed=17)
    freeze_protocol(tmp_path, protocol)
    freeze_protocol(tmp_path, protocol)
    before = (tmp_path / 'test_protocol.json').read_bytes()
    for change in (dict(protocol, seed=18), dict(protocol, checkpoints={'oracle': 'other-sha'})):
        with pytest.raises(ValueError, match='do not retune'):
            freeze_protocol(tmp_path, change)
    assert (tmp_path / 'test_protocol.json').read_bytes() == before
