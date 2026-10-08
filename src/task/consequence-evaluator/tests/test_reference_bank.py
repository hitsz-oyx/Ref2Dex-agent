"""Physical-bank provenance, frozen aggregation and causal contract tests."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / 'src'))
from consequence_evaluator import reference_bank
from consequence_evaluator.contracts import HAND_LINKS
from consequence_evaluator.data import sha
from consequence_evaluator.reference_progress import FeatureScale
from consequence_evaluator.temporal_phase import ENCODER_SCHEMA, LearnedReferenceProgress, TemporalPhaseEncoder
from consequence_evaluator.value_outcomes import RAW_SCHEMA


def test_selection_excludes_other_groups_and_uses_recorded_order(tmp_path, monkeypatch):
    raw = dict(schema=RAW_SCHEMA, status='COMPLETED', seed=230, fps=30, units='m',
        rollout_kind='continuous', audit_only=False, horizon=24, execution_horizon=24, episodes=[])
    # Excluded records deliberately have no files: they must never be opened.
    raw['episodes'] += [dict(split='val', assigned_phase='clean', perturbation_tick=-1),
                        dict(split='train', assigned_phase='contact', perturbation_tick=51)]
    poses = np.tile(np.eye(4), (543, 1, 1))
    packet = dict(action=np.zeros((542, 18)), object_pose=poses,
        hand_keypoints=np.zeros((543, 11, 3)), timestamps=np.arange(543) / 30,
        residual_plan=np.zeros((542, 24, 18)), plan_known=np.ones(542, dtype=bool))
    for i in range(10):
        file = tmp_path / ('e%d.npz' % i); diag = tmp_path / ('d%d.npz' % i)
        np.savez_compressed(file, **packet); np.savez_compressed(diag, qualification=i != 1)
        raw['episodes'].append(dict(episode='e%d' % i, split='train', split_group='source_seed:230',
            assigned_phase='clean', perturbation_tick=-1, task='airplane', motion='s3_airplane_lift',
            steps=542, path=file.name, diagnostics=diag.name, sha256=sha(file), diagnostics_sha256=sha(diag)))
    monkeypatch.setattr(reference_bank, 'task_trace', lambda p, d: dict(task_success=bool(d['qualification'])))
    members, packets, frozen = reference_bank.select_physical_references(tmp_path, raw)
    assert [m['episode'] for m in members] == ['e0', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8']
    assert len(packets) == 8
    assert str(tmp_path / 'e9.npz') not in frozen
    raw['seed'] = 261
    with pytest.raises(ValueError, match='train230'):
        reference_bank.select_physical_references(tmp_path, raw)


def test_bank_rejects_changed_hash_and_ambiguous_member_identity(tmp_path):
    file = tmp_path / 'reference.npz'
    np.savez_compressed(file, object_pose=np.tile(np.eye(4), (8, 543, 1, 1)),
        hand_keypoints=np.zeros((8, 543, 11, 3)), timestamps=np.tile(np.arange(543) / 30, (8, 1)))
    meta = dict(schema=reference_bank.BANK_SCHEMA, status='COMPLETED', origin=reference_bank.BANK_ORIGIN,
        hand_links=list(HAND_LINKS), reference_sha256=sha(file), source_seed=230,
        aggregation='fixed_uniform_mean', members=[dict(episode='e%d' % i) for i in range(8)])
    views, height = reference_bank.load_reference_features(tmp_path, meta)
    assert len(views) == 8 and views[0].shape == (543, 90) and height.shape == (543,)
    meta['members'][-1] = meta['members'][0]
    with pytest.raises(ValueError, match='eight-member'):
        reference_bank.load_reference_features(tmp_path, meta)
    meta['reference_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='fixed hash'):
        reference_bank.load_reference_features(tmp_path, meta)


def test_all_member_priors_are_causal_and_aggregation_is_fixed():
    # Tiny CPU contract smoke; all real fitting and label inference use GPU.
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    refs = [rng.normal(size=(70, 90)) for _ in range(2)]
    model = TemporalPhaseEncoder()
    bundle = dict(schema=ENCODER_SCHEMA, model=model.state_dict(), standardizer=FeatureScale(refs).dictionary())
    matcher = LearnedReferenceProgress(refs, bundle, 'cpu')
    actual = refs[0].copy(); changed = actual.copy(); changed[41:] += 100
    full = matcher.align(actual); other = matcher.align(changed); prefix = matcher.align(actual[:41])
    assert np.allclose(full['progress'], full['member_progress'].mean(axis=1), atol=1e-14)
    assert np.allclose(full['distribution'].sum(axis=1), 1)
    assert np.allclose(full['distribution'][:41], other['distribution'][:41], atol=1e-12)
    assert np.allclose(full['member_progress'][:41], prefix['member_progress'], atol=1e-12)
    assert np.max(np.abs(np.diff(full['progress']))) <= 4 / 69 + 1e-10
