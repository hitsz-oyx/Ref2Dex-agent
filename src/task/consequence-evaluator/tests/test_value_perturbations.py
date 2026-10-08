"""Pre-action geometry, empirical bank bounds, and full-plan execution checks."""
from pathlib import Path
import sys
import numpy as np
import pytest

TASK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.collection import Perturbations,PhaseAssignments
from consequence_evaluator.value_perturbations import (ValuePhaseTracker,VALUE_PHASES,
    BANK_SCHEMA,COUPLED,error_chunk,sample_bank)
from consequence_evaluator.value_outcomes import validate_plan_execution


def test_empirical_candidates_preserve_signs_and_ignore_overwritten_channels():
    errors=np.full((22,18),.3);errors[:,6]=-.1
    chunk=error_chunk(errors,1.5)
    assert chunk.shape==(24,18) and np.abs(chunk).max()<=.2+1e-6
    assert np.abs(chunk[:,:3]).max()<=.05+1e-6 and np.all(chunk[:,COUPLED]==0)
    assert (chunk[1:-1,6]<0).all() and np.all(chunk[[0,-1]]==0)
    bank=dict(schema=BANK_SCHEMA,source_split='train',phase='place',chunks=[chunk.tolist()])
    samples,ids=sample_bank(bank,6,231)
    assert samples.shape==(6,24,18) and not ids.any()
    with pytest.raises(ValueError,match='train-derived'):
        sample_bank(dict(bank,source_split='test'),6,231)
    bad=chunk.copy();bad[3,7]=.01
    with pytest.raises(ValueError,match='coupled'):
        sample_bank(dict(bank,chunks=[bad.tolist()]),6,231)


def test_place_trigger_requires_geometry_and_prior45unsupported_held_frames():
    tracker=ValuePhaseTracker([55,55,55])
    active=np.ones(3,bool);initial=np.zeros(3)
    for tick in range(56):
        # First is held; second is elevated but resting on table; third far hand.
        code=tracker.measure(np.full(3,.05),initial,np.array([.002,.002,.02]),
                             np.array([.05,0.,.05]),np.ones(3),tick,active)
        if tick==44:assert code[0]==3  # 44actual measured states
        if tick==45:assert code[0]==4
    assert code.tolist()==[5,2,0]
    assert tracker.completed.tolist()==[True,False,False]


def test_empirical_placing_executes_once_and_known_plan_matches_request():
    assignments=PhaseAssignments(231,phases=('place',),phase_names=VALUE_PHASES).assign(np.zeros(6))
    assert sorted(assignments.tolist())==[0,0,0,6,6,6]
    chunk=error_chunk(np.full((24,18),.04))
    p=Perturbations(6,231,assignment=assignments,phase_names=VALUE_PHASES,
                    chunks=np.stack([chunk]*6))
    plans=[];actions=[]
    for tick in range(60):
        phase=np.full(6,5 if tick>=10 else 4,np.int64)
        action,_,_=p.apply(np.zeros((6,18)),np.ones(6),np.ones(6,bool),np.zeros(6),
                           tick,np.full(6,100-tick),np.ones(6,bool),phase_code=phase)
        plan,known=p.known_plan(tick);plans.append(plan);actions.append(action)
        if tick<10:assert not known[assignments==6].any()
    i=int(np.flatnonzero(assignments==6)[0]);actions=np.asarray(actions)
    assert p.started[i]==10
    assert np.array_equal(plans[10][i],actions[10:34,i])
    assert not actions[34:,i].any() and not actions[:,assignments==0].any()
    short=Perturbations(6,231,assignment=assignments,phase_names=VALUE_PHASES)
    short.apply(np.zeros((6,18)),np.ones(6),np.ones(6,bool),np.zeros(6),10,
                np.full(6,23),np.ones(6,bool),phase_code=np.full(6,5,np.int64))
    assert (short.started==-1).all()


def test_stronger_registered_plan_keeps_requests_separate_from_native_clipping():
    chunk=error_chunk(np.full((24,18),.4),4.,residual_bound=.5)
    bank=dict(schema=BANK_SCHEMA,source_split='train',phase='contact',chunks=[chunk.tolist()],
              limits=dict(residual=.5))
    chunks,_=sample_bank(bank,1,251)
    with pytest.raises(ValueError,match='bounded'):
        Perturbations(1,251,assignment=np.array([2]),chunks=chunks)
    p=Perturbations(1,251,assignment=np.array([2]),chunks=chunks,residual_bound=.5)
    plans=[];actions=[]
    for tick in range(30):
        action,_,info=p.apply(np.full((1,18),.9),np.zeros(1),np.zeros(1,bool),np.zeros(1),
            tick,np.full(1,60-tick),np.ones(1,bool),phase_code=np.ones(1,np.int64))
        plan,known=p.known_plan(tick);plans.append(plan[0]);actions.append(action[0])
        if tick==12:assert info['clipped'].any() and info['residual'].max()>.4
    packet=dict(action=np.asarray(actions),residual_plan=np.asarray(plans),plan_known=np.array([False]+[True]*29))
    with pytest.raises(ValueError,match='invalid'):
        validate_plan_execution(packet,dict(perturbation_tick=1))
    validate_plan_execution(packet,dict(perturbation_tick=1),residual_bound=.5)
    assert np.array_equal(packet['residual_plan'][1],np.asarray([x[0] for x in packet['residual_plan'][1:25]]))
