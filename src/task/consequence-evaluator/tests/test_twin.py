"""CPU contracts for exact full-state twin branches."""
import numpy as np
import pytest

from consequence_evaluator.contracts import K
from consequence_evaluator.twin import (TwinBranch, capture_native_snapshot,
                                        capture_snapshot, replay_provenance,
                                        validate_pair)


REQUIRED = ('_root_states', '_dof_state', '_rigid_body_state', '_contact_forces',
            '_tar_contact_forces', 'obs_buf', '_hist_obs', 'progress_buf', 'data_id',
            'start_times', '_curr_obs', 'contact_reset', '_terminate_buf',
            'controller_state', 'rnn_state', 'observation', 'scalars', 'reset_ids')

RNG = dict(python=('state',), numpy=('MT19937',), torch_cpu=np.zeros(4, dtype='uint8'),
           torch_cuda=[])


def provenance(tick=40):
    return dict(prefix_hash='a' * 64, prefix_steps=tick, prefix_state_count=tick + 1,
                prefix_action_hash='b' * 64,
                prefix_action_count=tick, fresh_simulator=True, initial_frame_count=0,
                final_frame_count=tick, replay_max_abs_error=0., physics_properties_hash='c' * 64,
                history_contract_hash='d' * 64, controller_identity_hash='e' * 64, physics_dt=1/30)


def snapshot(pair='p0'):
    state = {name: np.zeros((2, 3), dtype='float32') for name in REQUIRED}
    state.update(controller_state={'policy': np.zeros(1)}, rnn_state=np.zeros(1),
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
            return 5

    task.gym = Gym()
    task.sim = object()
    task.dr_randomizations = {}
    task.projtype = 'None'
    task._motion_sampler = None
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
        physics_dt=1 / 30)
    snap = capture_native_snapshot(
        'native-p0', 5, task, controller_state={'policy': np.zeros(1)},
        rnn_state=np.zeros(1), observation=np.zeros(6), scalars={'dt': 1 / 30},
        reset_ids={'default': np.zeros(1)}, history=np.zeros((4, 6)),
        object_pose=np.eye(4, dtype='float32'), hand_keypoints=np.zeros((11, 3)),
        torch_module=TorchStub, replay=replay)
    assert snap.tick == 5
    assert snap.replay_provenance['prefix_action_count'] == 5
    assert snap.state_hash == snap.state_hash
    assert set(snap.rng) >= {'python', 'numpy', 'torch_cpu', 'torch_cuda'}


@pytest.mark.parametrize('states, actions', [
    ([None, None], np.zeros((1, 18), dtype='float32')),
    (['bad', 'bad'], np.zeros((1, 18), dtype='float32')),
    ([{}, {}], np.zeros((1, 18), dtype='float32')),
])
def test_replay_provenance_rejects_non_numeric_prefix_trace(states, actions):
    with pytest.raises(ValueError, match='finite'):
        replay_provenance(states, actions, replay_max_abs_error=0,
                          physics_properties={}, history_contract={},
                          controller_identity={}, physics_dt=1 / 30)


def test_replay_provenance_requires_native_action_shape():
    with pytest.raises(ValueError, match='shape'):
        replay_provenance(np.zeros((2, 1)), np.zeros((1, 17)),
                          replay_max_abs_error=0, physics_properties={},
                          history_contract={}, controller_identity={}, physics_dt=1 / 30)
