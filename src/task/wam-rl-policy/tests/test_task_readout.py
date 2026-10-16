import numpy as np
import pytest
import torch
from wam_rl_policy.task_readout import (finite_returns, object_split, transition_targets,
                                        TaskReadout, raw_tokens)


def test_returns_are_action_aligned_and_terminal_bootstrap_zero():
    reward=np.array([[1.,2.],[3.,4.]],np.float32)
    reset=np.array([[False,False],[False,False],[True,True]])
    np.testing.assert_allclose(finite_returns(reward,reset,.5),[[2.5,4.],[3.,4.]])
    reset[1,0]=True
    with pytest.raises(ValueError):finite_returns(reward,reset)


def test_split_keeps_all_repeats_of_object_together():
    rows=[dict(object_code=code,split='train') for code in ['d','a','c','b','a','d']]
    fit,held=object_split(rows)
    assert held==['a'] and fit.tolist()==[True,False,True,True,False,True]
    rows[0]['split']='dev'
    with pytest.raises(ValueError):object_split(rows)


def test_transition_labels_use_current_frame_and_goal():
    obj=np.zeros((40,1,7));obj[:,0,0]=np.arange(40)*.001
    hand=np.zeros((40,1,11,3));hand[:,:,:,0]=obj[:,:,0,None]+.03
    goal=np.zeros((40,1,3));goal[:,:,0]=.1;goal[20:]=100
    data=dict(object=obj,hand=hand,goal=goal,reset=np.zeros((40,1),bool))
    # Row-vector world x becomes local negative y under this orientation.
    rotation=np.array([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]])
    label=transition_targets(data,3,rotation)
    np.testing.assert_allclose(label,[[0,-.016,0,.081,.03,0,-.032,0,.065,.03]],atol=1e-7)
    data['reset'][34]=True
    with pytest.raises(ValueError):transition_targets(data,3,rotation)


def test_masked_tokens_do_not_affect_prediction():
    torch.manual_seed(7)
    model=TaskReadout(8,24).eval()
    x=torch.randn(2,8);tokens=torch.randn(2,9,24)
    valid=torch.ones(2,9,dtype=torch.bool);valid[:,-2:]=False
    original=model(x,tokens,valid)
    tokens[:,-2:]=10000
    changed=model(x,tokens,valid)
    for a,b in zip(original,changed):torch.testing.assert_close(a,b)
    with pytest.raises(ValueError):model(x,tokens,torch.zeros_like(valid))


def test_raw_tokens_preserve_only_observed_fields_and_hand_identity():
    batch=dict(xyz=torch.randn(2,5,3),features=torch.randn(2,5,18),
        start_hand=torch.randn(2,2,11,3),start_hand_history=torch.randn(2,2,11,9),
        start_object_points=torch.randn(2,32,3),goal_object_delta=torch.randn(2,32,3),
        point_valid=torch.ones(2,5,dtype=torch.bool),start_hand_valid=torch.ones(2,2,11,dtype=torch.bool),
        object_goal_valid=torch.ones(2,32,dtype=torch.bool))
    tokens,mask=raw_tokens(batch)
    assert tokens.shape==(2,59,24) and mask.all()
    torch.testing.assert_close(tokens[:,:5,:3],batch['xyz'])
    torch.testing.assert_close(tokens[:,5:27,3:12],batch['start_hand_history'].flatten(1,2))
    assert tokens[0,5,13]!=tokens[0,6,13] and tokens[0,5,12]!=tokens[0,16,12]
    batch['action_target']=torch.randn(2,8,24)*100
    batch['scene_target']=torch.randn(2,1,24,32,3)*100
    torch.testing.assert_close(raw_tokens(batch)[0],tokens)


def test_forecast_arm_requires_forecast_and_backpropagates_to_readout():
    model=TaskReadout(8,24,forecast_width=12)
    x=torch.randn(2,8);tokens=torch.randn(2,9,24);valid=torch.ones(2,9,dtype=torch.bool)
    with pytest.raises(ValueError):model(x,tokens,valid)
    value,physical=model(x,tokens,valid,torch.randn(2,12))
    (value.square().mean()+physical.square().mean()).backward()
    assert model.attention.in_proj_weight.grad.abs().sum()>0
    assert model.forecast.weight.grad.abs().sum()>0
