"""Honest condition deletion must preserve labels and task intent."""
import pickle
from pathlib import Path
import numpy as np
import pytest
import torch
from ptv3_shared_wam_v6.goal_policy import UNKNOWN_GOAL_POLICY,GOAL_KEYS,apply_goal_input_policy,assert_unknown_inputs
from ptv3_shared_wam_v6.events import UNKNOWN_EVENT_POLICY,apply_event_input_policy
from ptv3_shared_wam_v6.online_runtime import online


def test_unknown_erases_nan_goals_without_mutating_supervision_or_caller():
    source=dict(goal_object_delta=np.full((32,3),np.nan,np.float32),goal_object_points=np.ones((32,3),np.float32),object_goal_valid=np.ones(32,bool),trajectory_valid=np.ones((24,2,21),bool),scene_target=np.ones((1,24,32,3),np.float32))
    actual=apply_goal_input_policy(source,UNKNOWN_GOAL_POLICY)
    assert source['object_goal_valid'].all() and np.isnan(source['goal_object_delta']).all()
    for key in GOAL_KEYS:assert not actual[key].any() and actual[key].dtype==source[key].dtype
    for key in ('trajectory_valid','scene_target'):assert actual[key] is source[key]


def test_same_constructor_numpy_and_torch_is_bit_exact():
    source=dict(goal_object_delta=np.ones((3,32,3),np.float32),goal_object_points=np.ones((3,32,3),np.float32),object_goal_valid=np.ones((3,32),bool))
    a=apply_goal_input_policy(source,UNKNOWN_GOAL_POLICY);b=apply_goal_input_policy({k:torch.from_numpy(v) for k,v in source.items()},UNKNOWN_GOAL_POLICY)
    for k in GOAL_KEYS:assert a[k].tobytes()==b[k].numpy().tobytes()


def test_unknown_is_not_valid_zero_delta():
    b=dict(goal_object_delta=torch.zeros(2,32,3),goal_object_points=torch.zeros(2,32,3),object_goal_valid=torch.ones(2,32,dtype=torch.bool),event_type=torch.zeros(2,dtype=torch.long),event_delta_steps=torch.zeros(2),event_valid=torch.zeros(2,dtype=torch.bool))
    with pytest.raises(AssertionError):assert_unknown_inputs(b)
    assert_unknown_inputs(apply_goal_input_policy(b,UNKNOWN_GOAL_POLICY))


def test_online_goal_and_event_do_not_change_any_final_input():
    geo=dict(surface=np.zeros((2048,3),np.float32),normals=np.zeros((2048,3),np.float32),query=np.ones((32,3),np.float32))
    args=(geo,np.tile(np.eye(4),(4,1,1)),np.zeros((4,2,21,3)),np.zeros((4,2,778,3)),np.ones((2,21),bool),np.zeros((2,778),bool),np.zeros((2,778),np.int64))
    goal=np.eye(4);goal[:3,3]=[1,2,3];event=dict(event_type=2,event_valid=True,event_delta_steps=10.,event_hz=30)
    kwargs=dict(goal_input_policy=UNKNOWN_GOAL_POLICY,event_input_policy=UNKNOWN_EVENT_POLICY)
    left=online(*args,goal,event,**kwargs);right=online(*args,None,None,**kwargs)
    assert all(left[k].tobytes()==right[k].tobytes() for k in left)
    assert event['event_valid'] and np.array_equal(goal[:3,3],[1,2,3])


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError):apply_goal_input_policy({},'invented-goal')
