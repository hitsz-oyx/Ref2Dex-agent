import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1]/'tools/audit/prepare_wam_multitakeover.py'
spec = importlib.util.spec_from_file_location('multi', path)
multi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi)


def source():
    d = dict(expert_applied=np.zeros((128, 2), bool), expert_clipped=np.zeros((128, 2), bool),
             applied=np.zeros((128, 4, 18)), expert_action=np.zeros((128, 2, 18)),
             pd_targets=np.zeros((128, 4, 18)), expert_PD=np.zeros((128, 2, 18)))
    d['expert_applied'][16:, 1] = True
    return d


def test_only_complete_own_expert_futures_admitted():
    np.testing.assert_array_equal(multi.own_recovery_rows(source(), 16, 128), np.arange(16, 105))


@pytest.mark.parametrize('bad', ['mixed_control', 'clipped', 'borrowed_action', 'borrowed_PD'])
def test_invalid_future_or_current_label_rejected(bad):
    d = copy.deepcopy(source())
    if bad == 'mixed_control':
        d['expert_applied'][32, 1] = False
    elif bad == 'clipped':
        d['expert_clipped'][32, 1] = True
    elif bad == 'borrowed_action':
        d['applied'][32, 3, 0] = 1
    else:
        d['pd_targets'][32, 3, 0] = 1
    with pytest.raises(ValueError):
        multi.own_recovery_rows(d, 16, 128)
