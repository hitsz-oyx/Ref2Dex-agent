import copy
import json
import pytest
from ptv3_shared_wam_v2.native_execution import validate_robot_packet
from ptv3_shared_wam_v2.paths import SCHEMA
from ptv3_shared_wam.contracts import GOAL_ENCODING, TASK_CONDITIONING


def packet(tmp_path):
    controller = dict(offset=[0.] * 18, scale=[1.] * 18)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(schema='ref2dex.rl660-corpus.v1',
        status='COMPLETED', training_allowed=True, native_controller=controller)))
    return dict(schema=SCHEMA, stage='robot', config=dict(schema=SCHEMA, horizon=24,
        goal_encoding=GOAL_ENCODING, task_conditioning=TASK_CONDITIONING,
        action_representation='wrist-geometry-continuous-lift.v1'),
        provenance=dict(source=str(tmp_path)), model={'forecaster.x': 1, 'retargeter.x': 2})


def test_controller_bound_to_exact_source_without_mutating_checkpoint(tmp_path):
    original = packet(tmp_path)
    before = copy.deepcopy(original)
    loaded = validate_robot_packet(original)
    assert original == before
    assert loaded['config']['native_controller']['scale'] == [1.] * 18
    loaded['config']['native_controller']['scale'][0] = 2
    with pytest.raises(ValueError, match='mismatch'):
        validate_robot_packet(loaded)


@pytest.mark.parametrize('field,value', [('schema','ref2dex.ptv3-shared-wam.v1'),
    ('stage','pretrain'), ('model', {'retargeter.x': 2})])
def test_incompatible_checkpoint_rejected(tmp_path, field, value):
    data = packet(tmp_path)
    data[field] = value
    with pytest.raises(ValueError):
        validate_robot_packet(data)
