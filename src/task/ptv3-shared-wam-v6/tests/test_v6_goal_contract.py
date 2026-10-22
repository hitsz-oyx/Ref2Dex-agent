"""Targeted deployment-path checks for the v6 scheduler waypoint contract."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(ROOT / "src/task/wam-rl-policy-v3/src"))

from ptv3_shared_wam_v6.online_runtime import (  # noqa: E402
    GOAL_SEMANTICS,
    OnlineV6Runtime,
    _prepare_goal_inputs,
    online,
)


def _online_inputs():
    poses = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
    poses[:, :3, 3] = [1., 2., 3.]
    joints = np.zeros((4, 2, 21, 3), np.float32)
    joint_valid = np.zeros((2, 21), bool)
    joint_valid[0, :11] = True
    geo = dict(
        surface=np.zeros((2048, 3), np.float32),
        normals=np.tile([0., 0., 1.], (2048, 1)),
        query=np.arange(96, dtype=np.float64).reshape(32, 3) / 100.,
    )
    goal = poses[-1].copy()
    goal[:3, 3] += [.01, -.02, .08]
    return geo, poses, joints, joint_valid, goal


def test_v6_helper_is_bitwise_equivalent_to_v4_helper():
    from wam_rl_policy_v3.wam_v4_runtime import _prepare_goal_inputs as v4_prepare

    flow = np.arange(96, dtype=np.float64).reshape(1, 32, 3) / 100.
    flow[0, 7, 1] = np.nan
    raw = np.full((1, 32, 3), [.01, -.02, .08], dtype=np.float64)
    raw[0, 11, 2] = np.inf
    centers = np.zeros((1, 3), dtype=np.float64)
    rotations = np.broadcast_to(np.eye(3), (1, 3, 3)).copy()
    ours = _prepare_goal_inputs(
        flow, centers, rotations, goal_delta=raw,
        goal_semantics=GOAL_SEMANTICS)
    reference = v4_prepare(
        flow, centers, rotations, goal_delta=raw,
        goal_semantics=GOAL_SEMANTICS)
    assert np.array_equal(ours[0], reference[0])
    assert np.array_equal(ours[1], reference[1])


@pytest.mark.parametrize("goal", [None, "nan"])
def test_online_missing_and_nonfinite_goal_zero_fill_per_query(goal):
    geo, poses, joints, joint_valid, valid_goal = _online_inputs()
    if goal == "nan":
        goal = valid_goal.copy()
        goal[0, 1] = np.nan
    sample = online(
        geo, poses, joints, np.zeros((4, 2, 778, 3), np.float32),
        joint_valid, np.zeros((2, 778), bool), np.zeros((2, 778), np.int64),
        goal)
    assert sample["goal_object_delta"].shape == (32, 3)
    assert np.isfinite(sample["goal_object_delta"]).all()
    assert np.isfinite(sample["goal_object_points"]).all()
    assert not sample["object_goal_valid"].any()
    np.testing.assert_array_equal(sample["goal_object_delta"], np.zeros((32, 3)))
    np.testing.assert_array_equal(sample["goal_object_points"], np.zeros((32, 3)))


def test_online_nonfinite_homogeneous_row_is_also_invalid():
    geo, poses, joints, joint_valid, goal = _online_inputs()
    goal[3, 0] = np.nan
    sample = online(
        geo, poses, joints, np.zeros((4, 2, 778, 3), np.float32),
        joint_valid, np.zeros((2, 778), bool), np.zeros((2, 778), np.int64),
        goal)
    assert not sample["object_goal_valid"].any()
    np.testing.assert_array_equal(sample["goal_object_delta"], np.zeros((32, 3)))
    assert np.isfinite(sample["goal_object_points"]).all()


def test_online_valid_goal_is_scheduler_waypoint_and_wrong_semantics_rejected():
    geo, poses, joints, joint_valid, goal = _online_inputs()
    sample = online(
        geo, poses, joints, np.zeros((4, 2, 778, 3), np.float32),
        joint_valid, np.zeros((2, 778), bool), np.zeros((2, 778), np.int64),
        goal)
    assert sample["object_goal_valid"].all()
    np.testing.assert_allclose(sample["goal_object_delta"],
                               np.tile(goal[:3, 3] - poses[-1, :3, 3], (32, 1)))
    with pytest.raises(ValueError, match="spatial_path_waypoint"):
        online(
            geo, poses, joints, np.zeros((4, 2, 778, 3), np.float32),
            joint_valid, np.zeros((2, 778), bool), np.zeros((2, 778), np.int64),
            goal, goal_semantics="observed_t_plus_24_endpoint")


def test_predict_asserts_the_collated_model_goal_for_valid_missing_and_nonfinite():
    # Reuse the existing CPU model seam; this reaches predict(), collate(),
    # and SharedWAMV6 rather than stopping at the sample builder.
    from test_v6_executor_wiring import inputs, runtime

    rt = runtime()
    args = inputs()
    original = rt._trajectory
    captured = []

    def capture(batch):
        captured.append({key: value.detach().cpu().clone() for key, value in batch.items()})
        return original(batch)

    rt._trajectory = capture
    cases = [args[2], None, np.where(np.isfinite(args[2]), args[2], args[2]).copy()]
    cases[-1][0, 0, 0] = np.nan
    expected_valid = [True, False, False]
    for goal, all_valid in zip(cases, expected_valid):
        captured.clear()
        call_args = (args[0], args[1], goal, args[3], args[4])
        rt.predict(*call_args)
        batch = captured[0]
        assert batch["object_goal_valid"].dtype == torch.bool
        assert torch.isfinite(batch["goal_object_delta"]).all()
        invalid = ~batch["object_goal_valid"]
        assert torch.all(batch["goal_object_delta"][invalid] == 0)
        assert bool(batch["object_goal_valid"].all()) is all_valid

    with pytest.raises(ValueError, match="spatial_path_waypoint"):
        rt.predict(*args, goal_semantics="observed_t_plus_24_endpoint")
