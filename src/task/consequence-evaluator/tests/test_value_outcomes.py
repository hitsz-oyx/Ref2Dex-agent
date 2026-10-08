import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from consequence_evaluator.value_outcomes import task_trace,label_window,preference,validate_plan_execution


def fixture():
    steps=160
    height=np.zeros(steps+1,np.float32)
    height[20:110]=.05
    height[110:121]=np.linspace(.05,0,11)
    pose=np.tile(np.eye(4,dtype=np.float32),(steps+1,1,1))
    pose[:,2,3]=height+.8
    gap=np.full(steps+1,.003,np.float32);gap[118:]=.05
    velocity=np.zeros((steps+1,6),np.float32)
    velocity[110:120,2]=-.15
    packet=dict(action=np.zeros((steps,18),np.float32),object_pose=pose,
                residual_plan=np.zeros((steps,24,18),np.float32),plan_known=np.ones(steps,bool))
    diagnostics=dict(surface_gap=gap,contact_valid=np.r_[False,np.ones(steps,bool)],
                     contact=np.zeros(steps+1,bool),initial_height=.8,object_velocity=velocity,
                     reference_object_pose=pose.copy(),support_gap=height.copy(),table_footprint=np.ones(steps+1,bool))
    return packet,diagnostics


def test_controlled_return_is_success_including_late_windows():
    packet,diagnostics=fixture()
    trace=task_trace(packet,diagnostics)
    assert trace['task_success']
    assert label_window(trace,80)['success']==1
    assert label_window(trace,136)['success']==1
    assert label_window(trace,137) is None
    assert trace['stage'][-1]==5
    diagnostics['contact'][:]=True
    assert task_trace(packet,diagnostics)['task_success']


def test_early_drop_and_free_fall_to_table_are_failures():
    packet,diagnostics=fixture()
    packet['object_pose'][90:101,2,3]=.8
    diagnostics['surface_gap'][90:101]=.05
    assert not task_trace(packet,diagnostics)['task_success']
    packet,diagnostics=fixture()
    diagnostics['surface_gap'][116:119]=.08
    diagnostics['support_gap'][116:119]=.06
    diagnostics['object_velocity'][116:119,2]=-.5
    trace=task_trace(packet,diagnostics)
    assert not trace['task_success']
    assert trace['progress'].max()>0


def test_fall_to_floor_cannot_be_placed():
    packet,diagnostics=fixture()
    diagnostics['support_gap'][120:]=-.7
    assert not task_trace(packet,diagnostics)['task_success']


def test_full_requested_plan_must_match_execution_schedule():
    packet,_=fixture()
    validate_plan_execution(packet,dict(perturbation_tick=-1))
    packet['residual_plan'][0,1,0]=.03
    with pytest.raises(ValueError):
        validate_plan_execution(packet,dict(perturbation_tick=0))
    packet,_=fixture()
    with pytest.raises(ValueError,match='entire'):
        validate_plan_execution(packet,dict(perturbation_tick=150))


def test_success_then_progress_then_margin_with_abstention():
    first=dict(success=1,progress_summary=.1,margin=[0,0,-1])
    second=dict(success=0,progress_summary=1.,margin=[1,1,0])
    assert preference(first,second)==1
    assert preference(first,first)==0
    second.update(success=1)
    assert preference(first,second)==-1
