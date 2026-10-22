import numpy as np
import pytest

from ptv3_shared_wam_v6.scheduler_cache_identity import (
    GOAL_SEMANTICS, cache_identity, condition_tensor_hash,
    scheduler_conditions_for_ticks,
)
from wam_rl_policy_v3.goal_scheduler import OrderedGoalScheduler, update_schedulers


def _toy_inputs():
    poses = np.broadcast_to(np.eye(4), (8, 1, 4, 4)).copy()
    poses[:, 0, 0, 3] = np.arange(8, dtype=np.float64) * 0.005
    joints = np.zeros((8, 1, 21, 3), dtype=np.float64)
    joints[:, 0, 0, 2] = 0.01
    query = np.zeros((32, 3), dtype=np.float64)
    query[:, 0] = np.linspace(-.02, .02, 32)
    query[:, 1] = np.tile([-.01, .01], 16)
    query[:, 2] = np.repeat([-.01, .01], 16)
    surface = np.zeros((16, 3), dtype=np.float64)
    return dict(object_poses=poses, scheduler_reference_poses=poses, joints=joints), {"query": query, "surface": surface}


def test_identity_uses_source_and_composite_sample_id():
    condition = np.zeros((32, 3), np.float32)
    left = cache_identity(0, [0, 3], 3, condition)
    right = cache_identity(1, [0, 3], 3, condition)
    assert left["condition_sha256"] == right["condition_sha256"]
    assert left["identity"] != right["identity"]
    assert left["goal_semantics"] == GOAL_SEMANTICS


def test_training_replay_matches_the_inference_scheduler_path_bitwise():
    sequence, geometry = _toy_inputs()
    replayed = scheduler_conditions_for_ticks(sequence, geometry, 0, 0, [3, 5])
    from ptv3_shared_wam_v6.online_runtime import _goal_from_pose
    scheduler = OrderedGoalScheduler(sequence["object_poses"][:, 0], np.arange(8) / 30.,
                                     geometry["query"], interaction_points=geometry["surface"],
                                     goal_type="se3_pose")
    direct = {}
    hands = sequence["joints"][:, 0, (0, 1, 5, 9, 13, 17, 4, 8, 12, 16, 20)]
    for tick in range(6):
        decision = update_schedulers([scheduler], sequence["object_poses"][tick:tick+1, 0],
                                     np.asarray([tick / 30.]), hands[tick:tick+1],
                                     fast_geometry=True)[0]
        if tick in (3, 5):
            # The online v6 boundary converts the scheduler waypoint pose to
            # the same-ID current-object-frame tensor consumed by the model.
            online_delta, online_valid = _goal_from_pose(
                geometry["query"], np.linalg.inv(sequence["object_poses"][tick, 0]),
                decision["goal_pose"])
            assert online_valid.dtype == bool
            direct[tick] = np.asarray(online_delta[0], np.float32)
    assert np.array_equal(replayed[3], direct[3])
    assert np.array_equal(replayed[5], direct[5])
    assert condition_tensor_hash(replayed[3]) == condition_tensor_hash(direct[3])


def test_replay_uses_live_pose_instead_of_reference_pose():
    sequence, geometry = _toy_inputs()
    sequence['object_poses'] = sequence['object_poses'].copy()
    sequence['object_poses'][:, 0, 1, 3] = .004
    decisions = scheduler_conditions_for_ticks(sequence, geometry, 0, 0, [3], return_details=True)
    from ptv3_shared_wam_v6.online_runtime import _goal_from_pose
    expected, valid = _goal_from_pose(geometry['query'], np.linalg.inv(sequence['object_poses'][3, 0]),
                                     decisions[3]['goal_pose'])
    assert valid.all()
    assert np.array_equal(decisions[3]['goal_delta'], expected[0])
    assert np.all(decisions[3]['goal_delta'][:, 1] < 0)


def test_no_future_observation_reads_with_fixed_independent_instruction():
    sequence, geometry = _toy_inputs()
    tick = 3
    expected = scheduler_conditions_for_ticks(sequence, geometry, 0, 0, [tick])[tick]
    for mode in ('truncate', 'nan', 'guard'):
        changed = dict(sequence)
        for field in ('object_poses', 'joints'):
            value = sequence[field].copy()
            if mode == 'truncate':
                value = value[:tick+1]
            elif mode == 'nan':
                value[tick+1:] = np.nan
            else:
                class PrefixOnly:
                    def __init__(self, values): self.values = values
                    def __array__(self, *args, **kwargs):
                        raise AssertionError('whole-array observation read')
                    def __getitem__(self, index):
                        frame = index[0] if isinstance(index, tuple) else index
                        assert isinstance(frame, int) and 0 <= frame <= tick, 'future observation read'
                        return self.values[index]
                value = PrefixOnly(value)
            changed[field] = value
        actual = scheduler_conditions_for_ticks(changed, geometry, 0, 0, [tick])[tick]
        assert np.array_equal(actual, expected), mode


def test_missing_instruction_is_a_hard_error():
    sequence, geometry = _toy_inputs()
    del sequence['scheduler_reference_poses']
    with pytest.raises(ValueError, match='reference path absent'):
        scheduler_conditions_for_ticks(sequence, geometry, 0, 0, [3])


def test_identical_observation_prefix_does_not_determine_task_intent():
    # Both tasks have identical measured histories and identical reference
    # prefixes. The instruction after t can legitimately ask for opposite
    # motions. No observation-only causal producer can distinguish them.
    sequence, geometry = _toy_inputs()
    sequence['object_poses'] = np.broadcast_to(np.eye(4), (8, 1, 4, 4)).copy()
    outputs = []
    for sign in (-1, 1):
        instruction = sequence['object_poses'].copy()
        instruction[4:, 0, 0, 3] = sign * np.arange(1, 5) * .02
        task = dict(sequence, scheduler_reference_poses=instruction)
        outputs.append(scheduler_conditions_for_ticks(task, geometry, 0, 0, [3])[3])
    assert np.all(outputs[0][:, 0] < 0)
    assert np.all(outputs[1][:, 0] > 0)
    assert not np.array_equal(*outputs)
