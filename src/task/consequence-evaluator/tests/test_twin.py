"""CPU contracts for exact full-state twin branches."""
import random
import pickle
from pathlib import Path
import sys
from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from consequence_evaluator.contracts import K
from consequence_evaluator.twin import (TwinBranch, capture_native_snapshot,
                                        capture_snapshot, replay_provenance,
                                        restore_native_rng, validate_pair, fingerprint)


REQUIRED = ('_root_states', '_dof_state', '_rigid_body_state', '_contact_forces',
            '_tar_contact_forces', 'obs_buf', '_hist_obs', 'progress_buf', 'data_id',
            'start_times', '_curr_obs', 'contact_reset', '_terminate_buf',
            'controller_state', 'rnn_state', 'observation', 'scalars', 'reset_ids')

RNG = dict(python=random.getstate(), numpy=np.random.get_state(),
           torch_cpu=np.zeros(4, dtype='uint8'),
           torch_cuda=[])


def test_content_identity_ignores_string_and_dtype_object_sharing():
    value = 'a_long_noninterned_string_123456789'
    shared = {'a': value, 'b': value}
    separate = {'a': value, 'b': ''.join([value[:-1], value[-1]])}
    assert fingerprint(shared) == fingerprint(separate)
    tree = {'first': np.arange(4, dtype='float32'), 'second': np.arange(4, dtype='float32')}
    assert fingerprint(tree) == fingerprint(pickle.loads(pickle.dumps(tree, protocol=4)))


def test_serialized_native_snapshot_retains_branch_content_identity():
    snap = snapshot()
    first, second = branch(snap, 'a', .02), branch(snap, 'b', -.02)
    loaded = pickle.loads(pickle.dumps((snap, first, second), protocol=4))
    assert loaded[0].common_prefix_hash == snap.common_prefix_hash
    assert validate_pair(loaded[0], loaded[0], loaded[1], loaded[2])


def provenance(tick=40):
    return dict(prefix_hash='a' * 64, prefix_steps=tick, prefix_state_count=tick + 1,
                prefix_action_hash='b' * 64,
                prefix_action_count=tick, fresh_simulator=True, initial_frame_count=0,
                final_frame_count=tick, replay_max_abs_error=0., physics_properties_hash='c' * 64,
                history_contract_hash='d' * 64, controller_identity_hash='e' * 64,
                physics_dt=1/30, control_dt=1/30, sim_steps_per_control=1)


def snapshot(pair='p0'):
    state = {name: np.zeros((2, 3), dtype='float32') for name in REQUIRED}
    state.update(controller_state={'policy': np.zeros(1)},
                 rnn_state={'is_rnn': True, 'state': np.zeros(1)},
                 observation=np.zeros(6), scalars={'dt': 1/30}, reset_ids={'default': np.zeros(1)})
    return capture_snapshot(pair, 40, state, RNG, np.zeros((4, 6)),
                            np.eye(4, dtype='float32'), np.zeros((11, 3), dtype='float32'),
                            required_state_keys=REQUIRED, replay_provenance=provenance())


def branch(snap, branch_id, value):
    plan = np.zeros((K, 18), dtype='float32')
    plan[:, 3] = value
    actions = np.zeros((K, 18), dtype='float32')
    actions[:, 3] = value
    return TwinBranch(snap.state_hash, snap.pair_id, branch_id, plan,
                      actions,
                      np.tile(np.eye(4, dtype='float32'), (K + 1, 1, 1)),
                      np.zeros((K + 1, 11, 3), dtype='float32'),
                      np.zeros(K, dtype=bool), snap.common_prefix_hash)


def test_twin_requires_full_state_and_records_exact_prefix_identity():
    first = snapshot()
    second = snapshot()
    assert first.common_prefix_hash == second.common_prefix_hash
    result = validate_pair(first, second, branch(first, 'a', .02), branch(second, 'b', -.02))
    assert result['residual_plan_l2'] > .01 and result['actions'] == 24
    with pytest.raises(ValueError, match='missing native'):
        capture_snapshot('p', 0, {'_root_states': np.zeros(1)}, {}, np.zeros(1),
                         np.eye(4), np.zeros((11, 3)), required_state_keys=REQUIRED,
                         replay_provenance=provenance(0))


def test_twin_rejects_pose_only_or_same_plan_forks():
    first, second = snapshot(), snapshot()
    second = capture_snapshot('p0', 40, {**first.state, '_dof_state': first.state['_dof_state'] + 1},
                              first.rng, first.history, first.object_pose, first.hand_keypoints,
                              required_state_keys=REQUIRED, replay_provenance=provenance())
    with pytest.raises(ValueError, match='common prefix'):
        validate_pair(first, second, branch(first, 'a', .02), branch(second, 'b', -.02))
    same_a, same_b = branch(first, 'a', .02), branch(first, 'b', .02)
    with pytest.raises(ValueError, match='indistinguishable'):
        validate_pair(first, first, same_a, same_b)


def test_twin_rejects_wrong_branch_anchor():
    snap = snapshot()
    wrong = branch(snap, 'a', .02)
    wrong.object_poses[0, 0, 3] = .01
    with pytest.raises(ValueError, match='wrong object state'):
        validate_pair(snap, snap, wrong, branch(snap, 'b', -.02))


def test_twin_rejects_early_done():
    snap = snapshot()
    plan = np.zeros((K, 18), dtype='float32'); plan[:, 0] = .02
    done = np.zeros(K, dtype=bool); done[3] = True
    with pytest.raises(ValueError, match='terminated'):
        TwinBranch(snap.state_hash, snap.pair_id, 'a', plan, np.zeros((K, 18)),
                   np.tile(np.eye(4), (K + 1, 1, 1)), np.zeros((K + 1, 11, 3)), done,
                   snap.common_prefix_hash)


def test_twin_rejects_nonintegral_tick_and_out_of_range_actual_action():
    bad_provenance = provenance(); bad_provenance['prefix_steps'] = 40.0
    with pytest.raises(ValueError, match='provenance'):
        capture_snapshot('p', 40, snapshot().state, RNG, np.zeros(1), np.eye(4),
                         np.zeros((11, 3)), required_state_keys=REQUIRED,
                         replay_provenance=bad_provenance)
    with pytest.raises(ValueError, match='tick'):
        capture_snapshot('p', 3.7, snapshot().state, RNG, np.zeros(1), np.eye(4),
                         np.zeros((11, 3)), required_state_keys=REQUIRED,
                         replay_provenance=provenance(3))
    snap = snapshot()
    actions = np.zeros((K, 18), dtype='float32'); actions[0, 0] = 1.1
    with pytest.raises(ValueError, match='executed controls'):
        TwinBranch(snap.state_hash, snap.pair_id, 'a', np.zeros((K, 18)), actions,
                   np.tile(np.eye(4), (K + 1, 1, 1)),
                   np.zeros((K + 1, 11, 3)), np.zeros(K, dtype=bool), snap.common_prefix_hash)


def test_native_adapter_captures_full_task_and_fresh_prefix_provenance():
    class NativeTask:
        pass

    task = NativeTask()
    class Gym:
        @staticmethod
        def get_frame_count(sim):
            assert sim is not None
            return 10

    task.gym = Gym()
    task.sim = object()
    task.dr_randomizations = {}
    task.projtype = 'None'
    task._motion_sampler = None
    task.control_freq_inv = 2
    task.dt = 1 / 30
    task.sim_params = SimpleNamespace(dt=1 / 60)
    task._enable_early_termination = False
    task._adaptive_kappa_enabled = False
    task.rollout_length = 1200
    class StateInit(Enum):
        Start = 1
    task._state_init = StateInit.Start
    for name in REQUIRED:
        if name not in ('controller_state', 'rnn_state', 'observation', 'scalars', 'reset_ids'):
            setattr(task, name, np.zeros((2, 3), dtype='float32'))
    task._root_states[:, 0] = 0
    class TorchStub:
        class cuda:
            @staticmethod
            def is_available():
                return False

        @staticmethod
        def get_rng_state():
            return np.zeros(4, dtype='uint8')

    replay = replay_provenance(
        [np.zeros((2, 3), dtype='float32')] * 6,
        np.zeros((5, 18), dtype='float32'),
        replay_max_abs_error=2e-5, physics_properties={'dt': 1 / 30},
        history_contract={'shape': [4, 6]}, controller_identity={'sha256': 'actor'},
        physics_dt=1 / 60, sim_steps_per_control=2)
    snap = capture_native_snapshot(
        'native-p0', 5, task, controller_state={'policy': np.zeros(1)},
        rnn_state=np.zeros(1), observation=np.zeros(6), scalars={'dt': 1 / 30},
        reset_ids={'default': np.zeros(1)}, history=np.zeros((4, 6)),
        object_pose=np.eye(4, dtype='float32'), hand_keypoints=np.zeros((11, 3)),
        torch_module=TorchStub, replay=replay, is_rnn=True)
    assert snap.tick == 5
    assert snap.replay_provenance['prefix_action_count'] == 5
    assert snap.replay_provenance['final_frame_count'] == 10
    assert snap.state_hash == snap.state_hash
    assert set(snap.rng) >= {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
    inventory = snap.state['native_scalar_inventory']
    assert inventory['_enable_early_termination'] is False
    assert inventory['_adaptive_kappa_enabled'] is False
    assert inventory['projtype'] == 'None'
    assert inventory['rollout_length'] == 1200
    assert inventory['_state_init'] == {
        'enum_type': StateInit.__module__ + '.' + StateInit.__qualname__,
        'name': 'Start', 'value': 1,
    }
    no_rnn = capture_native_snapshot(
        'native-p0-no-rnn', 5, task, controller_state={'policy': np.zeros(1)},
        rnn_state=None, observation=np.zeros(6), scalars={'dt': 1 / 30},
        reset_ids={'default': np.zeros(1)}, history=np.zeros((4, 6)),
        object_pose=np.eye(4, dtype='float32'), hand_keypoints=np.zeros((11, 3)),
        torch_module=TorchStub, replay=replay, is_rnn=False)
    assert no_rnn.state['rnn_state'] == {'is_rnn': False, 'state': None}
    task.control_freq_inv = 1
    with pytest.raises(ValueError, match='decimation'):
        capture_native_snapshot(
            'native-p0-bad-clock', 5, task, controller_state={'policy': np.zeros(1)},
            rnn_state=None, observation=np.zeros(6), scalars={'dt': 1 / 30},
            reset_ids={'default': np.zeros(1)}, history=np.zeros((4, 6)),
            object_pose=np.eye(4, dtype='float32'), hand_keypoints=np.zeros((11, 3)),
            torch_module=TorchStub, replay=replay, is_rnn=False)


def test_restore_native_rng_restores_cpu_streams_with_stub_torch():
    class TorchStub:
        uint8 = np.dtype('uint8')
        restored = None

        @staticmethod
        def tensor(value, dtype=None, device=None):
            assert dtype == np.dtype('uint8')
            assert device == 'cpu'
            return np.asarray(value, dtype='uint8')

        @staticmethod
        def set_rng_state(value):
            TorchStub.restored = value.copy()

        class cuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def device_count():
                return 0

    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        restore_native_rng(RNG, TorchStub)
        assert np.array_equal(TorchStub.restored, RNG['torch_cpu'])
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_restore_native_rng_rejects_cuda_topology_mismatch():
    class TorchStub:
        uint8 = np.dtype('uint8')

        @staticmethod
        def tensor(value, dtype=None, device=None):
            assert device == 'cpu'
            return np.asarray(value, dtype='uint8')

        @staticmethod
        def set_rng_state(value):
            pass

        class cuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def device_count():
                return 0

    bad = dict(RNG, torch_cuda=[np.zeros(4, dtype='uint8')],
               torch_cuda_device_count=1, torch_cuda_device_order=[0])
    with pytest.raises(ValueError, match='device count'):
        restore_native_rng(bad, TorchStub)


@pytest.mark.parametrize('states, actions', [
    ([None, None], np.zeros((1, 18), dtype='float32')),
    (['bad', 'bad'], np.zeros((1, 18), dtype='float32')),
    ([{}, {}], np.zeros((1, 18), dtype='float32')),
])
def test_replay_provenance_rejects_non_numeric_prefix_trace(states, actions):
    with pytest.raises(ValueError, match='finite'):
        replay_provenance(states, actions, replay_max_abs_error=0,
                          physics_properties={'dt': 1 / 30}, history_contract={'shape': [1]},
                          controller_identity={'id': 'test'}, physics_dt=1 / 30)


def test_replay_provenance_requires_native_action_shape():
    with pytest.raises(ValueError, match='shape'):
        replay_provenance(np.zeros((2, 1)), np.zeros((1, 17)),
                          replay_max_abs_error=0, physics_properties={'dt': 1 / 30},
                          history_contract={'shape': [1]}, controller_identity={'id': 'test'},
                          physics_dt=1 / 30)


def test_zero_step_replay_accepts_natural_empty_action_list():
    result = replay_provenance([np.zeros((2, 3), dtype='float32')], [],
                               replay_max_abs_error=0, physics_properties={'dt': 1 / 30},
                               history_contract={'shape': [1]}, controller_identity={'id': 'test'},
                               physics_dt=1 / 30)
    assert result['prefix_steps'] == 0 and result['prefix_state_count'] == 1


def test_zero_step_replay_rejects_explicit_wrong_width():
    with pytest.raises(ValueError, match='shape'):
        replay_provenance([np.zeros((2, 3), dtype='float32')], np.zeros((0, 17)),
                          replay_max_abs_error=0, physics_properties={'dt': 1 / 30},
                          history_contract={'shape': [1]}, controller_identity={'id': 'test'},
                          physics_dt=1 / 30)


@pytest.mark.parametrize('field', ['physics_properties', 'history_contract', 'controller_identity'])
def test_replay_provenance_rejects_missing_contract(field):
    kwargs = dict(physics_properties={'dt': 1 / 30}, history_contract={'shape': [1]},
                  controller_identity={'id': 'test'}, physics_dt=1 / 30)
    kwargs[field] = None
    with pytest.raises(ValueError, match='provenance'):
        replay_provenance([np.zeros((2, 3), dtype='float32')], [], replay_max_abs_error=0,
                          **kwargs)


def test_twin_rejects_unwrapped_rnn_state():
    state = snapshot().state
    state = dict(state, rnn_state=np.zeros(1))
    with pytest.raises(ValueError, match='explicit is_rnn'):
        capture_snapshot('legacy-rnn', 40, state, RNG, np.zeros((4, 6)),
                         np.eye(4), np.zeros((11, 3)), required_state_keys=REQUIRED,
                         replay_provenance=provenance())


@pytest.mark.parametrize('bad_rng', [
    {'python': None, 'numpy': None, 'torch_cpu': None, 'torch_cuda': []},
    {**RNG, 'torch_cpu': np.zeros(4, dtype='float32')},
    {**RNG, 'torch_cuda': [np.zeros(4, dtype='uint8')] ,
     'torch_cuda_device_count': 0},
])
def test_twin_rejects_malformed_rng_state(bad_rng):
    with pytest.raises(ValueError, match='RNG state'):
        capture_snapshot('bad-rng', 40, snapshot().state, bad_rng, np.zeros((4, 6)),
                         np.eye(4), np.zeros((11, 3)), required_state_keys=REQUIRED,
                         replay_provenance=provenance())


def test_twin_rejects_malformed_rnn_sentinel():
    snap = snapshot()
    state = dict(snap.state, rnn_state={'is_rnn': False, 'state': 123})
    with pytest.raises(ValueError, match='state=None'):
        capture_snapshot('bad-rnn', snap.tick, state, snap.rng, snap.history,
                         snap.object_pose, snap.hand_keypoints,
                         required_state_keys=REQUIRED, replay_provenance=provenance())
