import numpy as np
import pytest
import torch

from trajectory_policy.chunk_retargeter import ChunkRetargeter, chunk_windows, expert_chunk_rows
from trajectory_policy.retargeter_data import ACTIVE


def test_actual_control_chunk_is_aligned_with_own_next24_states():
    controls,envs=30,2
    obj=np.broadcast_to(np.eye(4,dtype=np.float32),(31,envs,4,4)).copy()
    hand=np.broadcast_to(np.arange(31,dtype=np.float32)[:,None,None,None],(31,envs,11,3)).copy()
    action=np.broadcast_to(np.arange(controls,dtype=np.float32)[:,None,None]/30,(controls,envs,18)).copy()
    data=dict(q=np.zeros((31,envs,18),np.float32),dq=np.zeros((31,envs,18),np.float32),
              hand=hand,obj=obj,velocity=np.zeros((31,envs,6),np.float32),action=action,applied=action.copy())
    state,tau,label=chunk_windows(data,[0,1])
    assert state.shape==(14,87) and tau.shape==(14,24,33) and label.shape==(14,8,12)
    np.testing.assert_array_equal(tau[0,:,0],np.arange(1,25))
    np.testing.assert_array_equal(label[0,:,0],np.arange(8,dtype=np.float32)/30)
    np.testing.assert_array_equal(label[2,:,0],np.arange(1,9,dtype=np.float32)/30)
    data['action'][0,0,0]=99
    with pytest.raises(ValueError):chunk_windows(data,[0,1])


def test_chunk_action_is_native_and_tau_remains_differentiable():
    torch.set_num_threads(1);torch.manual_seed(295)
    model=ChunkRetargeter().eval()
    with torch.no_grad():model.head.weight.normal_(0,.01)
    state=torch.randn(2,87);tau=torch.randn(2,24,33,requires_grad=True)
    raw=model(state,tau)
    assert raw.shape==(2,8,12)
    raw[:,0].square().sum().backward()
    assert tau.grad is not None and tau.grad.abs().sum()>0
    native=model.native_action(state,tau.detach())
    assert native.shape==(2,8,18)
    assert native.abs().max()<=1
    passive=[i for i in range(18) if i not in ACTIVE]
    assert native[...,passive].count_nonzero()==0


def test_clipped_or_unapplied_expert_cannot_supply_chunk_supervision():
    d=dict(applied=np.zeros((32,4,18)),expert_action=np.zeros((32,2,18)),
           expert_applied=np.ones((32,2),bool),expert_clipped=np.zeros((32,2),bool))
    d['expert_applied'][10,1]=False
    with pytest.raises(ValueError):expert_chunk_rows(d,8)
    d['expert_applied'][10,1]=True;d['expert_clipped'][10,1]=True
    with pytest.raises(ValueError):expert_chunk_rows(d,8)
