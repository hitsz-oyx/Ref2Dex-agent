"""Isaac-free regression checks for the native engineering runner."""
import copy
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

RUNNER = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/audit/native_twin_probe.py'))


def test_physics_enum_does_not_recurse_into_its_own_members():
    class BoundEnum:
        __members__ = {'ALL': None}
        name = 'ALL'

        def __int__(self):
            return 2

    value = BoundEnum()
    BoundEnum.ALL = value  # dir(value) exposes a self-reference, as pybind does.
    props = SimpleNamespace(contact_collection=value, substeps=2)
    result = RUNNER['normalize_properties'](props)
    assert result['contact_collection']['value'] == 2
    assert result['contact_collection']['name'] == 'ALL'
    assert result['substeps'] == 2


@pytest.mark.parametrize('field', ['residual_plan', 'actions', 'object_poses',
                                  'hand_keypoints', 'done', 'branch_states'])
def test_repeat_gate_detects_each_future_or_execution_change(field):
    first = {'snapshot': SimpleNamespace(common_prefix_hash='prefix'),
             'branch': SimpleNamespace(**{name: np.zeros(2) for name in
                ('residual_plan', 'actions', 'object_poses', 'hand_keypoints', 'done')}),
             'branch_states': [{'dof': np.zeros(2)}]}
    repeat = copy.deepcopy(first)
    assert all(RUNNER['repeat_checks'](first, repeat).values())
    if field == 'branch_states':
        repeat[field][0]['dof'][0] = 1
    else:
        getattr(repeat['branch'], field)[0] = 1
    assert not all(RUNNER['repeat_checks'](first, repeat).values())
