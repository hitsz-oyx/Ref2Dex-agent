"""Event-free policy boundaries and fail-closed launch decisions; no training."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

TASK = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('event_free_audit',
    TASK/'tools/audit/freeze_event_free_tau.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize('kind,delta,valid', [(0, 0., True), (1, 8., True),
    (2, 40., True), (2, float('nan'), False)])
def test_unknown_mask_preserves_every_non_event_input_and_target(kind, delta, valid):
    sample = dict(event_type=np.int64(kind), event_delta_steps=np.float32(delta),
                  event_valid=np.bool_(valid), event_hz=np.int64(30),
                  goal_object_delta=np.ones((32, 3), np.float32),
                  object_goal_valid=np.ones(32, bool),
                  trajectory_target=np.ones((24, 2, 21, 3), np.float32),
                  contact_target=np.ones((1, 24, 4), np.float32))
    raw_arrays = {key: value for key, value in sample.items() if key not in audit.EVENT_DTYPES}
    batch = audit.event_free_collate([sample]*8)
    features, mask = audit.event_features(batch)
    assert features.shape == (8, 5) and not features.any() and not mask.any()
    assert not ((batch['event_type'] == 0) & batch['event_valid']).any()
    assert batch['event_type'].dtype == torch.int64
    assert batch['event_delta_steps'].dtype == torch.float32
    assert batch['event_valid'].dtype == torch.bool
    for key, value in raw_arrays.items():
        np.testing.assert_array_equal(batch[key].numpy(), np.stack([value]*8))
        assert audit.event_free_sample(sample)[key] is value
    assert sample['event_valid'] == valid  # no mutation of the original sample


def test_deployment_adapter_reaches_real_predict_and_masks_event_encoder():
    from test_v6_executor_wiring import runtime, inputs
    rt = runtime()
    # Do not let zero-initialized event weights make the mask check vacuous.
    with torch.no_grad():
        rt.model.event_goal.encoder[-1].bias.fill_(7.)
    captures = []
    original = rt._trajectory
    def capture(batch):
        captures.append({key: value.clone() for key, value in batch.items()})
        return original(batch)
    rt._trajectory = capture
    args = inputs()
    # Show that default production calls are still historical; recipe metadata
    # alone cannot mask deployment events. This remains a failed launch gate.
    rt.predict(*args)
    assert all(batch['event_valid'].all() for batch in captures)
    goals = [batch['goal_object_delta'].clone() for batch in captures]
    captures.clear()
    result = audit.event_free_predict(rt, *args)
    assert result.shape == (2, 24, 11, 3)
    for i, batch in enumerate(captures):
        assert not batch['event_valid'].any()
        assert not audit.event_features(batch)[0].any()
        assert torch.equal(batch['goal_object_delta'], goals[i])
        with torch.inference_mode():
            assert not rt.model.event_goal(batch).any()


def test_missing_event_slot_is_not_silently_broadcast():
    with pytest.raises(ValueError, match='one event slot'):
        audit.event_free_predict(None, [1, 2], None, None, [{}], None)


@pytest.mark.parametrize('gate', ['shared_event_policy_installed_in_production',
    'curated_training_reader_connected', 'scheduler_goal_cache_available',
    'gpu_allocation_available', 'disk_available'])
def test_each_launch_gate_independently_blocks(gate):
    probe = dict(production_training_wired=True, production_runtime_policy_enforced=True,
                 reader_windows=6000)
    receipt = dict(counts=dict(scheduler_eligible_windows=6000))
    resources = dict(status='MEASURED', available_unreserved_gpus=list(range(5)),
                     filesystem_free_bytes=100*2**30)
    if gate == 'shared_event_policy_installed_in_production':
        probe['production_runtime_policy_enforced'] = False
    elif gate == 'curated_training_reader_connected': probe['reader_windows'] = 207579
    elif gate == 'scheduler_goal_cache_available': receipt['counts']['scheduler_eligible_windows'] = 0
    elif gate == 'gpu_allocation_available': resources['available_unreserved_gpus'] = []
    else: resources['filesystem_free_bytes'] = 21*2**30
    result = audit.launch_gates(probe, receipt, resources, 2*2**30)
    assert result['status'] == 'BLOCKED_PREFLIGHT'
    assert not result['gates'][gate]
    assert sum(not passed for passed in result['gates'].values()) == 1
    assert result['launch'] is None and result['training_updates'] == 0


def test_unknown_prototype_cannot_release_production_training():
    result = audit.launch_gates(dict(production_training_wired=False,
        production_runtime_policy_enforced=False, reader_windows=6000),
        dict(counts=dict(scheduler_eligible_windows=6000)),
        dict(status='MEASURED', available_unreserved_gpus=list(range(8)),
             filesystem_free_bytes=100*2**30), 10*2**30)
    assert result['status'] == 'BLOCKED_PREFLIGHT'


def test_tuple_identity_rejects_duplicate_or_dropped_windows():
    rows = [dict(source_id=1, sequence_id=0, slot=0, tick=i,
                 split='train', admitted=True) for i in range(6000)]
    tuples = audit.tuple_rows(dict(selection=dict(train_window_rows=rows)))
    assert len(tuples) == 6000 and tuples[3] == [1, [0, 0, 3], 3]
    with pytest.raises(ValueError, match='6000 distinct'):
        audit.tuple_rows(dict(selection=dict(train_window_rows=rows[:-1])))
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match='6000 distinct'):
        audit.tuple_rows(dict(selection=dict(train_window_rows=rows)))
