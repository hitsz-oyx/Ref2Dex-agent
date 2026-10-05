"""Meaningful decision-time geometry / exact-fit contracts."""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src/task/cm-interaction-oracle/src'))
from geometric_consequence import (NominalSurfaceActions, physics_baseline, pca_fit,
    pca_apply, ridge_fit, ridge_predict)


def test_physics_baseline_world_axes_and_current_interaction():
    before = torch.zeros(2,72)
    before[:,6] = 1
    before[:,7] = .3
    before[:,12] = 1
    before[:,48] = 2
    before[:,66] = .01
    before[:,71] = 1
    output = physics_baseline(before,.25)
    assert torch.allclose(output[:,0],torch.tensor([.075,.075]))
    assert torch.allclose(output[:,5],torch.tensor([.25,.25]))
    assert output[:,6:12].abs().max()==0
    assert torch.allclose(output[:,12],torch.log(torch.tensor([3.,3.])))
    assert torch.allclose(output[:,20],before[:,66])
    assert (output[:,-1]==1).all()
    before[:,12]=7*torch.pi
    wrapped=physics_baseline(before,1.)
    assert torch.allclose(wrapped[:,3:6].norm(dim=-1),torch.tensor([torch.pi,torch.pi]),atol=1e-5)


def test_pca_and_ridge_ignore_heldout_targets_and_geometry():
    torch.manual_seed(42)
    x=torch.randn(18,6)
    y=torch.randn(18,3)
    ids=torch.arange(12)
    norm=pca_fit(x,ids,4)
    changed=x.clone();changed[12:]+=100
    again=pca_fit(changed,ids,4)
    assert torch.equal(norm['projection'],again['projection'])
    changed_y=y.clone();changed_y[12:]+=100
    model=ridge_fit(pca_apply(x,norm),y,ids)
    other=ridge_fit(pca_apply(changed,again),changed_y,ids)
    assert torch.equal(model['weight'],other['weight'])


def test_exact_centered_ridge_has_unpenalized_intercept():
    torch.manual_seed(43)
    x=torch.randn(12,3); y=torch.full((12,2),7.)
    output=ridge_predict(x,ridge_fit(x,y,torch.arange(12)))
    assert torch.allclose(output,y,atol=1e-6)


def test_nominal_geometry_uses_root_and_no_future_fields():
    # Tiny deterministic mesh/FK engineering smoke; CPU avoids GPU startup.
    from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose
    from src.task.CmResidual.v118_planner import QUERY_LINKS
    from intervention import per_finger_residuals
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    bridge=NominalSurfaceActions(urdf,'cpu',seed=42)
    q=torch.zeros(1,18);q[:,6:]=.2
    hand_root=torch.zeros(1,13);hand_root[:,6]=1;hand_root[:,0]=.5
    hand_root[:,3:7]=torch.tensor([[0.,0.,.70710678,.70710678]])
    world=dexplore_root_pose(hand_root)[:,None,None]@bridge.kinematics.forward(q[:,None])
    before=torch.zeros(1,72);before[:,6]=1
    pose=world[:,0,0]
    from scipy.spatial.transform import Rotation
    quat=torch.as_tensor(Rotation.from_matrix(pose[:,:3,:3].numpy()).as_quat(),dtype=q.dtype)
    base_pose=torch.cat((pose[:,:3,3],quat),-1)
    tips=[QUERY_LINKS.index(name+'_tip') for name in ('index','middle','pinky','ring','thumb')]
    packet=dict(history=q[:,None].expand(-1,10,-1),base_action=torch.zeros(1,18),
        delta=per_finger_residuals(.2),hand_root=hand_root,before=before,
        before_hand_base_pose=base_pose,before_fingertip_positions=world[:,0,tips,:3,3])
    result=bridge.build(packet)
    assert result['fingertip_error_max_m']<1e-6
    assert result['flow'][:,0].abs().max()==0
    assert result['nominal_flow'][:,0].abs().max()>0
    assert result['flow'][:,1].abs().max()>0
    # All necessary fields are decision-time. Future states are absent.
