import numpy as np

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
    return dict(object_poses=poses, joints=joints), {"query": query, "surface": surface}


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
            direct[tick] = np.asarray(decision["goal_delta"], np.float32)
    assert np.array_equal(replayed[3], direct[3])
    assert np.array_equal(replayed[5], direct[5])
    assert condition_tensor_hash(replayed[3]) == condition_tensor_hash(direct[3])
