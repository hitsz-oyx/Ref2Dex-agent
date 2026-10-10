"""Pure-position FK must match the actual state FK and its fitting gradient."""
from pathlib import Path
import torch
from consequence_evaluator.contracts import HAND_LINKS
from consequence_evaluator.reference_motion import NATIVE_DOF_NAMES
from consequence_evaluator.reset_kinematics import ResetKinematics


def test_positions_match_full_state_fk_and_joint_gradient():
    urdf=Path(__file__).resolve().parents[4]/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    fk=ResetKinematics(urdf,NATIVE_DOF_NAMES,HAND_LINKS,'cpu')
    generator=torch.Generator().manual_seed(29)
    q=torch.randn(5,18,generator=generator).requires_grad_()
    dq=torch.randn(5,18,generator=generator)
    root=torch.randn(5,13,generator=generator)
    root[:,3:7]/=root[:,3:7].norm(dim=-1,keepdim=True)
    expected=fk.states(q,dq,root)[:,:,:3]
    actual=fk.positions(q,root)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    weights=torch.randn(expected.shape,generator=generator)
    old_gradient=torch.autograd.grad((expected*weights).sum(),q)[0]
    new_gradient=torch.autograd.grad((actual*weights).sum(),q)[0]
    torch.testing.assert_close(new_gradient,old_gradient,rtol=1e-6,atol=2e-7)
