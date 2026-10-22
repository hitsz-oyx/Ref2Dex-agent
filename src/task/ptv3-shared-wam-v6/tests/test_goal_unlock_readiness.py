"""Evidence probes must not manufacture an unknown training batch or intent."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

TASK=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('goal_readiness',TASK/'tools/audit/audit_goal_unlock_readiness.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def goals(valid=False):
    return dict(goal_object_delta=torch.zeros(2,32,3),goal_object_points=torch.zeros(2,32,3),
                object_goal_valid=torch.full((2,32),valid))


def test_capture_stops_before_forward_and_does_not_rewrite_batch():
    class Consumer(nn.Module):
        def forward(self,batch):raise AssertionError('forward must not run')
    model=Consumer();batch=goals(True)
    actual=audit.capture_boundary(model,lambda:model(batch))
    assert all(audit.equal_bits(actual,batch).values())
    assert batch['object_goal_valid'].all()
    assert not model._forward_pre_hooks


def test_pre_model_rejection_does_not_create_a_training_unknown_tensor():
    def reject():raise ValueError('scheduler binding absent')
    model=nn.Identity()
    with pytest.raises(ValueError,match='scheduler binding absent'):
        audit.capture_boundary(model,reject)
    assert not model._forward_pre_hooks


def test_zero_valid_is_not_unknown_even_when_delta_values_match():
    valid,unknown=goals(True),goals(False)
    bits=audit.equal_bits(valid,unknown)
    assert bits['goal_object_delta']
    assert not bits['object_goal_valid']


def test_bit_comparison_checks_signed_zero_and_dtype():
    first,second=goals(),goals()
    second['goal_object_delta'].fill_(-0.)
    assert not audit.equal_bits(first,second)['goal_object_delta']
    second=goals();second['goal_object_delta']=second['goal_object_delta'].double()
    assert not audit.equal_bits(first,second)['goal_object_delta']


def test_native_shadow_parameters_do_not_become_generic_displacements():
    mothers=[dict(sequence_id=4,windows=30,layer_paths=['native'],
                  native_goal_position_evidence=dict(first=[0,0,.9]))]
    layers=[dict(path='native',fields=dict(goal={}),metadata={})]
    result=audit.count_rules(mothers,layers,dict(mothers=1064,total_windows=6000,certified_legal=0),
                            dict(rows=[dict(sequence_id=4)]))
    native=next(r for r in result if r['id']=='B4-native-Shadow-constructor')
    fixed=next(r for r in result if r['id']=='B1-observed-reset-plus-fixed-offset')
    assert native['parameter_mothers']==1 and native['parameter_windows']==30
    assert native['certified_full_waypoint_windows']==0
    assert fixed['literal_parameter_candidate_mothers']==0
    assert all(not row['implemented'] for row in result)


def test_hand_and_geometry_fields_do_not_supply_rule_parameters():
    result=audit.count_rules([dict(sequence_id=1,windows=100,layer_paths=['x'])],
        [dict(path='x',fields=dict(hand={},object_poses={},normals={},surface_parts={}),metadata={})],
        dict(mothers=1064,total_windows=6000,certified_legal=0),dict(rows=[]))
    assert all(row.get('literal_parameter_candidate_mothers',0)==0 for row in result)


def test_recorded_parameter_candidate_does_not_imply_reset_binding():
    result=audit.count_rules([dict(sequence_id=1,windows=100,layer_paths=['x'])],
        [dict(path='x',fields={},metadata=dict(meta=dict(config=dict(approach_distance=.1))))],
        dict(mothers=1064,total_windows=6000,certified_legal=0),dict(rows=[]))
    approach=next(row for row in result if row['id']=='B2-approach-direction')
    assert approach['literal_parameter_candidate_mothers']==1
    assert approach['certified_full_waypoint_mothers']==0
