"""Surface-relative sign/frame, source-only preprocessing and future whitelist."""
import sys
import os
from pathlib import Path
import torch
import pytest

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src')]
from contact_innovation import surface_action,surface_state,action_normalize_fit,bounded_design,contact_inputs
from geometric_consequence import NominalSurfaceActions


def test_surface_sign_identity_and_rigid_frame_invariance():
    obj=torch.zeros(1,1,3);normal=torch.tensor([[[1.,0.,0.]]])
    points=torch.zeros(1,15,120,3);points[...,0]=.01
    points[:,1,:24,0]-=.002;points[:,2,:24,0]+=.002
    value=surface_action(points,obj,normal)
    assert value[:,0].abs().max()==0 and value[0,1,0]<0 and value[0,2,0]>0
    assert value[0,1,1]>0 and value[0,2,1]==0
    assert value[0,1,3]>0 and value[0,2,3]<0
    assert value[0,1,6:30].abs().max()==0
    rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    translation=torch.tensor([.12,-.03,.05])
    moved=surface_action(points@rotation+translation,obj@rotation+translation,normal@rotation)
    assert torch.allclose(value,moved,atol=1e-5)
    assert torch.allclose(surface_state(points[:,0],obj,normal),surface_state(points[:,0]@rotation+translation,obj@rotation+translation,normal@rotation),atol=1e-5)


def test_action_normalizer_source_only_and_bounded_design():
    torch.manual_seed(55);a=torch.randn(5,15,32);train=torch.arange(3)
    norm=action_normalize_fit(a,train);a[3:]+=10000
    again=action_normalize_fit(a,train)
    assert torch.equal(norm['mean'],again['mean']) and torch.equal(norm['scale'],again['scale'])
    h=torch.randn(5,4)*100
    x=bounded_design(h,a[:,1],norm)
    assert x[:,:4].abs().max()<=8 and x[:,4:].abs().max()<=5


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_contact_input_ignores_future_and_predicted_wrist(device):
    if device.startswith('cuda') and os.environ.get('REF2DEX_GPU_CONTRACT_TEST')!='1':
        pytest.skip('GPU contract requires explicitly selected idle device')
    bridge=NominalSurfaceActions(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf',device,seed=56)
    q=torch.zeros(1,15,18);q[...,6:]=.2;q[:,1,6]+=.05
    root=torch.zeros(1,13);root[:,6]=1
    history=torch.zeros(1,1,18);history[:,:,:]=q[:,:1]
    p=dict(history=history,before=root.clone(),hand_root=root.clone(),trajectory=torch.ones(1,32,72))
    from execution_geometry import endpoint_flows
    # Keep collector p on CPU, candidate q/geometry on selected model device.
    q=q.to(device)
    raw,_=endpoint_flows(p,bridge,{'points':torch.zeros(1,120,3,device=device)},q[:,:1])
    g=dict(points=raw[:,0],obj_points=torch.zeros(1,1,3,device=device),obj_normals=torch.tensor([[[1.,0.,0.]]],device=device))
    first=contact_inputs(p,bridge,g,q)
    p['trajectory']*=100;other=q.clone();other[:,:,:6]+=10
    second=contact_inputs(p,bridge,g,other)
    assert all(torch.equal(x,y) for x,y in zip(first,second))
