"""Leakage guards for source OOF consequences consumed by a task learner."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT/'src/task/cm-interaction-oracle/src')]
from oracle_flow_task import validate_folds, assemble_crossfit


def test_environment_overlap_rejected_even_when_windows_disjoint():
    train, test = np.arange(6), np.arange(6, 8)
    clusters = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    folds = np.array([0, 0, 1, 1, 2, 2, -1, -1])
    records = validate_folds(train, test, clusters, folds)
    assert sum(len(r['hold']) for r in records.values()) == 6
    folds[1] = 1
    with pytest.raises(AssertionError): validate_folds(train, test, clusters, folds)
    folds[1] = 0; clusters[7] = 0
    with pytest.raises(AssertionError): validate_folds(train, test, clusters, folds)


def test_source_features_are_held_predictions_not_in_sample_or_full_fit():
    partitions = {'fold0': dict(fit=np.array([1, 2]), hold=np.array([0])),
                  'fold1': dict(fit=np.array([0, 2]), hold=np.array([1])),
                  'fold2': dict(fit=np.array([0, 1]), hold=np.array([2]))}
    full = torch.full((4, 2), 999.)
    predictions = {name: torch.full_like(full, -777.) for name in partitions}
    for j, name in enumerate(partitions): predictions[name][j] = j+1
    merged = assemble_crossfit(np.arange(3), np.array([3]), partitions, predictions, full)
    assert torch.equal(merged[:, 0], torch.tensor([1., 2., 3., 999.]))
    predictions['fold1'][1] = float('nan')
    with pytest.raises(AssertionError): assemble_crossfit(np.arange(3), np.array([3]), partitions, predictions, full)


def test_crossfit_cannot_fill_outer_test_from_a_source_fold():
    partitions = {'fold0': dict(fit=np.array([1]), hold=np.array([0, 2])),
                  'fold1': dict(fit=np.array([0]), hold=np.array([1]))}
    full = torch.zeros(3, 2); predictions = {name: full for name in partitions}
    with pytest.raises(AssertionError): assemble_crossfit(np.array([0, 1]), np.array([2]), partitions, predictions, full)
