"""Regression tests for the offline multi-zero Value audit."""
from pathlib import Path
import importlib.util

import pytest


TASK = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'audit_gpu_group_gt_values', TASK / 'tools/audit/audit_gpu_group_gt_values.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_multi_zero_role_names_and_y_noise_do_not_index_four_role_list():
    roles, noise = MODULE.summarize_value_noise(
        [.10, .11, .07, .04, .105, .095],
        ['baseline', 'zero_repeat', 'positive', 'negative'],
        [0, 1, 4, 5])
    assert roles == ['baseline', 'zero_repeat', 'positive', 'negative',
                     'zero_role_4', 'zero_role_5']
    assert len(noise['zero_pair_deltas']) == 6
    assert noise['candidate_deltas_vs_baseline'][0]['candidate'] == 'finger_positive'
    assert noise['candidate_deltas_vs_baseline'][1]['candidate'] == 'finger_negative'


def test_value_noise_rejects_candidate_as_zero_role():
    with pytest.raises(ValueError, match='role contract'):
        MODULE.summarize_value_noise(
            [.10, .11, .07, .04],
            ['baseline', 'zero_repeat', 'positive', 'negative'],
            [0, 2])


def test_swapped_role_map_keeps_baseline_and_candidate_labels_attached_to_envs():
    roles, noise = MODULE.summarize_value_noise(
        [.07, .04, .10, .11, .105, .095],
        ['baseline', 'zero_repeat', 'positive', 'negative'],
        [2, 3, 4, 5],
        {'baseline': 2, 'zero_repeat': 3, 'positive': 0, 'negative': 1})
    assert roles[:4] == ['positive', 'negative', 'baseline', 'zero_repeat']
    assert noise['candidate_deltas_vs_baseline'][0]['role'] == 0
    assert noise['candidate_deltas_vs_baseline'][0]['delta_y'] == pytest.approx(-0.03)
    assert noise['baseline_relative_zero_deltas'][0]['role'] == 3
