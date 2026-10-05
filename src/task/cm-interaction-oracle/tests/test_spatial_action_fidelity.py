"""Causal intrinsic frame and source-only decoder contracts."""
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT), str(ROOT/'src/task/cm-interaction-oracle/src')]
from spatial_action_fidelity import intrinsic_finger_flow, fit_decoder
from geometric_consequence import NominalSurfaceActions


def test_intrinsic_flow_removes_shared_wrist_and_ignores_future():
    bridge=NominalSurfaceActions(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf', 'cpu')
    q=torch.zeros(1,15,18);q[...,2]=.5;q[...,6:]=.2;q[:,1,6]+=.3
    root=torch.zeros(1,13);root[:,6]=1
    p=dict(hand_root=root,before=root.clone())
    g=dict(points=torch.zeros(1,120,3))
    a=intrinsic_finger_flow(p,bridge,g,q)
    changed=q.clone();changed[...,0:3]+=torch.tensor([.4,-.1,.5]);changed[...,3:6]+=torch.tensor([3.5,-2.,1.])
    # Candidate-dependent wrist motion is deliberately removed as well.
    changed[:,1,3]+=.4
    p['native_q']=torch.randn(1,32,18)*100
    b=intrinsic_finger_flow(p,bridge,g,changed)
    assert torch.allclose(a,b,atol=2e-7)
    assert a[:,0].abs().max()==0 and a[:,1].abs().max()>.001


def test_decoder_fit_is_invariant_to_held_features_and_labels():
    torch.manual_seed(42)
    x=torch.randn(8,15,12);y=torch.randn(8,15,12);ids=torch.arange(6)
    _,a=fit_decoder(x,y,ids)
    x[6:]*=100;y[6:]+=100
    _,b=fit_decoder(x,y,ids)
    for key in a['norm']:assert torch.equal(a['norm'][key],b['norm'][key])
    for key in ('weight','x_mean','y_mean'):assert torch.equal(a['model'][key],b['model'][key])
