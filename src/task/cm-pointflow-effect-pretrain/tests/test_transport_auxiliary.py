"""Future hands/targets cannot cross the rigid-transport forward boundary."""
import copy
from pathlib import Path
import sys

import pytest
import torch

TASK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK/'src'))
from oakink_wm.transport_auxiliary import (HISTORY_INPUTS,history_inputs,
    validate_transport_source,validate_auxiliary_config,classify_auxiliary,initial_primary_deltas)


def test_auxiliary_forward_has_no_future_hands_effects_or_outcome_metadata():
    current={k:torch.ones(2) for k in HISTORY_INPUTS}
    full=dict(current,action=torch.full((2,24,22,9),float('nan')),
              action_valid=torch.ones(2,24,22,dtype=torch.bool),
              effect=torch.full((2,1,24,4,4),float('nan')),quality='future_success')
    allowed=history_inputs(full)
    assert set(allowed)==set(current)
    assert all(allowed[k] is current[k] for k in current)
    del full['object_features']
    with pytest.raises(ValueError,match='current/history'):history_inputs(full)


def test_auxiliary_source_rejects_dynamic_or_weak_hand_semantics():
    meta=dict(source='contactpose',schema='ref2dex.native-wm30.v1',status='COMPLETED',
              training_allowed=True,fps=30,history=4,horizon=24,units='m',hand_order=['right','left'],
              hand_label='fixed_articulation_native_21_joints_with_per_frame_rigid_transforms',
              supervision_scope='rigid_grasp_transport; no dynamic finger or time-varying contact GT')
    descriptor=dict(name='contactpose',kind='native')
    validate_transport_source(meta,descriptor)
    for changed in (dict(source='epic_contact'),dict(training_allowed=False),dict(hand_label='dynamic_fingers'),dict(units='mm')):
        with pytest.raises(ValueError,match='provenance'):validate_transport_source(dict(meta,**changed),descriptor)


def test_auxiliary_probe_budget_and_predeclared_screen():
    config=dict(updates=300,main_batch=32,aux_batch=8,seconds=1800,validation_microbatch=2,
                auxiliary_weight=.05,learning_rate=1e-5,seed=227,aux_validation_samples=64)
    validate_auxiliary_config(config)
    for key,value in (('updates',501),('seconds',1801),('auxiliary_weight',float('nan')),('main_batch',True),('seed',400)):
        changed=copy.deepcopy(config);changed[key]=value
        with pytest.raises(ValueError):validate_auxiliary_config(changed)
    control=dict(main_macro_mm=30.,oakink2_mm=10.,transport_mm=40.)
    assert classify_auxiliary(control,dict(main_macro_mm=29.,oakink2_mm=10.2,transport_mm=35.))=='PROMISING'
    assert classify_auxiliary(control,dict(main_macro_mm=32.,oakink2_mm=10.,transport_mm=35.))=='UNPROMISING'
    assert classify_auxiliary(control,dict(main_macro_mm=30.,oakink2_mm=10.,transport_mm=40.))=='UNCLEAR'


def test_matching_primary_metrics_rejects_real_drift_but_allows_ancillary_acos_roundoff():
    first=dict(main_macro_mm=30.,oakink2_mm=10.,transport_mm=40.,
               main_metrics={s:{'model/anchor/cat0/h24/point_epe':.03,'rotation':.167731255}
                             for s in ('oakink2','grab','arctic')})
    second=copy.deepcopy(first);second['main_metrics']['grab']['rotation']-=1.52e-5
    assert not any(initial_primary_deltas(first,second).values())
    second['main_macro_mm']+=.002
    with pytest.raises(ValueError,match='1micrometre'):initial_primary_deltas(first,second)
