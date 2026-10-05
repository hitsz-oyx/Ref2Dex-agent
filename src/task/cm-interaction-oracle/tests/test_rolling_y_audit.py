from pathlib import Path
import sys
import pytest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src')]
from rolling_y_audit import short_y,rolling_y,failure_events,pair_signal,OFFSETS,DENSE_OFFSETS
from consequence_sufficiency import readouts


def test_sufficient_fields_reproduce_original_risk_and_all_heads():
    torch.manual_seed(7)
    before=torch.randn(20,72);before[:,2]=torch.rand(20)*.06;before[:,71]=torch.arange(20)%2
    trajectory=torch.randn(20,32,72);trajectory[:,:,2]=torch.rand(20,32)*.07
    trajectory[:,:,71]=(torch.rand(20,32)>.4).float();rest=torch.zeros(20)
    _,_,_,expected,_=readouts(dict(before=before,trajectory=trajectory,rest_height=rest))
    actual,risk=short_y(before[:,2],before[:,71]>.5,trajectory[:,:,2],trajectory[:,:,71]>.5,rest)
    assert torch.equal(actual,expected)
    assert torch.equal(risk,(before[:,2]>=.03)&(before[:,71]>.5))


def test_step89_event_is_censored_at_last_8step_query_but_visible_at_dense57():
    height=torch.full((2,90),.04);height[0,88:]=.019;pair=torch.ones_like(height,dtype=torch.bool)
    y,_=rolling_y(height,pair,torch.zeros(2),torch.full((2,),.04),torch.ones(2,dtype=torch.bool))
    assert torch.equal(y[0],y[1])
    dense,_=rolling_y(height,pair,torch.zeros(2),torch.full((2,),.04),torch.ones(2,dtype=torch.bool),DENSE_OFFSETS)
    assert dense[0,56,6]==0 and dense[0,57,6]==1 and dense[1,57,6]==0
    with pytest.raises(ValueError):rolling_y(height,pair,torch.zeros(2),torch.full((2,),.04),torch.ones(2,dtype=torch.bool),(64,))


def test_contact_loss_spans_local_step8_and_current_risk_is_recomputed():
    height=torch.full((2,32),.04);pair=torch.ones(2,32,dtype=torch.bool);pair[:,3:9]=False
    y,risk=short_y(torch.tensor([.04,.01]),torch.ones(2,dtype=torch.bool),height,pair,torch.zeros(2))
    assert risk.tolist()==[True,False] and y[:,2].tolist()==[1,0]
    assert y[:,6].tolist()==[0,0] and y[:,7].tolist()==[1,1]


def test_qualification_failure_does_not_invent_drop_and_prequalification_event_remains():
    height=np.full(90,.04);pair=np.ones(90,dtype=bool)
    assert failure_events(height,pair,0,-1)['first_raw_failure_step'] is None
    height[4]=.01;height[88]=.01
    info=failure_events(height,pair,0,50)
    assert info['first_raw_failure_step']==5 and info['first_postqualification_drop_step']==89


def test_wrong_direction_late_queries_and_low_risk_do_not_count_as_warning():
    offsets=(0,8,16,24);gap=(0,.1,-.2,.5)
    info=pair_signal(gap,(True,True,True,True),offsets,30)
    assert info['first_correct_separation']=={'offset':8,'lead_steps':22}
    assert info['delayed_signal'] is None  # later inversion; lastquery only6step lead
    info=pair_signal((0,.1,.2,.3),(True,False,True,True),offsets,40)
    assert info['delayed_signal']=={'offset':16,'lead_steps':24}
