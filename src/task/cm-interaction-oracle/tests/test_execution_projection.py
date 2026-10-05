"""Continuous wrist forecasts must preserve their simulator angle branch."""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT/'src/task/cm-interaction-oracle/src')]
from execution_projection import project_execution, continuous_mask
from geometric_consequence import NominalSurfaceActions


def test_continuous_wrist_projection_preserves_fk_and_unwrapped_angle():
    urdf = ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    bridge = NominalSurfaceActions(urdf, 'cpu')
    q = torch.zeros(1, 1, 18)
    q[..., 2] = .5
    q[..., 3:6] = torch.tensor([torch.pi+.3, -torch.pi-.4, torch.pi+.2])
    q[..., 6:] = .2
    continuous = continuous_mask(urdf, 'cpu')
    assert continuous.nonzero().flatten().tolist() == [3, 4, 5]
    projected = project_execution(q, bridge.lower, bridge.upper, continuous)
    assert torch.equal(projected[..., 3:6], q[..., 3:6])
    assert torch.allclose(bridge.kinematics.forward(projected), bridge.kinematics.forward(q), atol=1e-6)
    shifted = q.clone()
    shifted[..., 3:6] -= 2*torch.pi
    assert torch.allclose(bridge.kinematics.forward(shifted), bridge.kinematics.forward(q), atol=1e-6)
