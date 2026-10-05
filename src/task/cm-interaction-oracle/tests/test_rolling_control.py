from pathlib import Path
import sys
import copy
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src')]
from rolling_control import candidate_scores,mixed_plan,execution_z,control_gate


def panel(k):
    return dict(candidate=k,post_window=32,initial_fingerprint='cold',simulation_contract={},
        model_fingerprint='actor',rms_fingerprint='rms',rolling_offset=8,
        triggers=torch.tensor([42]),rest_height=torch.zeros(1),motion_id=torch.zeros(1),
        start_frame=torch.zeros(1),delta=torch.zeros(7,18),before=torch.zeros(1,72),
        history=torch.zeros(1,10,139),actor_obs=torch.zeros(1,2),hand_root=torch.zeros(1,13),
        full_world_prefix_errors=torch.zeros(1,7),valid_steps=torch.ones(1,32,dtype=torch.bool),
        height=torch.full((1,32),.04),pair=torch.ones(1,32,dtype=torch.bool))


def test_current_state_and_complete_window_are_required():
    panels=[panel(k) for k in range(7)]; rows=torch.tensor([0])
    y,s=candidate_scores(panels,rows)
    assert y.shape==(1,7,8) and torch.equal(s,torch.full((1,7),1.25))
    broken=copy.deepcopy(panels);broken[2]['before'][0,2]=.001
    with pytest.raises(ValueError,match='current H'):candidate_scores(broken,rows)
    broken=copy.deepcopy(panels);broken[2]['valid_steps'][0,-1]=False
    with pytest.raises(ValueError,match='incomplete'):candidate_scores(broken,rows)
    broken=copy.deepcopy(panels);broken[2]['full_world_prefix_errors'][0,0]=.001
    with pytest.raises(ValueError,match='replay'):candidate_scores(broken,rows)


def test_plan_uses_same_state_scores_and_baseline_ties():
    scores=torch.zeros(2,7);scores[1,4]=1
    assert mixed_plan(scores,torch.tensor([1,3]),5).tolist()==[0,0,0,4,0]
    scores[1,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):mixed_plan(scores,torch.tensor([1,3]),5)


def test_z_comes_from_contiguous_executed_path_and_checks_late_drop():
    trace=dict(after_physical=torch.zeros(100,2,72),done=torch.zeros(100,2,dtype=torch.bool))
    trace['after_physical'][:,:,2]=.04;trace['after_physical'][:,:,71]=1
    trace['after_physical'][94,1,2]=.01
    z,_=execution_z(trace,torch.tensor([0,1]),5,torch.zeros(2))
    assert z.tolist()==[True,False]
    with pytest.raises(ValueError,match='actual90'):execution_z(trace,torch.tensor([0,1]),11,torch.zeros(2))


def test_gate_counts_harms_and_rescues_without_reusing_old_upper_bound():
    baseline=np.r_[np.ones(23),np.zeros(9)];rolling=baseline.copy()
    rolling[0]=0;rolling[23:27]=1
    result=control_gate(baseline,rolling,np.arange(32)%2)
    assert result['rolling_count']==26 and result['rescued']==4 and result['harmed']==1
    assert result['status']=='UNPROMISING'  # uncertain paired gain, not a formal claim
