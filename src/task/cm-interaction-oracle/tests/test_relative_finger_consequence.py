"""Anatomical identity and action-centered prediction contracts."""
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src')]
from relative_finger_consequence import finger_summary,RelativeFingerHead


def test_summary_per_point_invariance_preserves_finger_identity_and_sign():
    flow=torch.zeros(1,15,120,3);flow[:,1,:24,0]=.02;flow[:,2,:24,0]=-.02
    value=finger_summary(flow)
    assert not value[:,0].any() and value.shape==(1,15,32)
    assert torch.allclose(value[0,1:3,0],torch.tensor([1.,-1.]),atol=1e-6)
    assert torch.allclose(value[0,1:3,15],torch.ones(2),atol=1e-6)
    shuffled=flow.reshape(1,15,5,24,3)[:,:,:,torch.randperm(24)].reshape_as(flow)
    assert torch.equal(value,finger_summary(shuffled))
    swapped=flow.reshape(1,15,5,24,3)[:,:,[1,0,2,3,4]].reshape_as(flow)
    other=finger_summary(swapped)
    assert other[0,1,0]==0 and torch.allclose(other[0,1,3],torch.tensor(1.),atol=1e-6)


def test_centered_head_removes_state_only_change_and_is_candidate_equivariant():
    torch.manual_seed(44);model=RelativeFingerHead(4,True)
    h=torch.randn(2,4);blank=torch.zeros(2,15,32)
    assert model(h,blank).abs().max()<1e-6
    assert model(h*100,blank).abs().max()<1e-6
    a=torch.randn(2,15,32);perm=torch.randperm(15)
    value=model(h,a)
    assert value.mean(1).abs().max()<1e-6
    assert torch.allclose(model(h,a[:,perm]),value[:,perm],atol=1e-6)


def test_factual_centered_loss_uses_other_recipient_candidates():
    torch.manual_seed(45);model=RelativeFingerHead(3,True)
    h=torch.randn(1,3);a=torch.randn(1,15,32,requires_grad=True)
    model(h,a)[0,1].square().sum().backward()
    assert a.grad[0,0].abs().sum()>0 and a.grad[0,2].abs().sum()>0
    plain=RelativeFingerHead(3,False);plain.load_state_dict(model.state_dict())
    other=a.detach().clone().requires_grad_(True)
    plain(h,other)[0,1].square().sum().backward()
    assert other.grad[0,0].abs().sum()==0 and other.grad[0,1].abs().sum()>0
