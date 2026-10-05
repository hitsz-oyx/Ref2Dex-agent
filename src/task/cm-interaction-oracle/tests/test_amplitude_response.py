import importlib.util
from pathlib import Path
import torch

SPEC = importlib.util.spec_from_file_location("amplitude_contract", Path(__file__).resolve().parents[1] / "src/intervention.py")
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def test_joint_amplitude_draw_and_scaled_live_clipping():
    arms, alpha = contract.decode_amplitude_assignment(torch.arange(21), [1, 2, 4])
    assert arms.tolist() == list(range(7))*3
    assert alpha.tolist() == [1.]*7+[2.]*7+[4.]*7
    base = torch.zeros(21, 18)
    result = contract.apply_feedback_residual(base, arms, torch.zeros(21), torch.full((21,), 8), torch.zeros(21, dtype=torch.bool), contract.residuals(), alpha)
    assert torch.allclose(result, contract.residuals()[arms]*alpha[:, None])
    base[-2, 6] = .9
    result = contract.apply_feedback_residual(base, arms, torch.zeros(21), torch.full((21,), 8), torch.zeros(21, dtype=torch.bool), contract.residuals(), alpha)
    assert result[-2, 6] == 1


def test_force_direction_is_preserved_when_norm_is_equal():
    force = torch.tensor([[0., 0., 5.], [4., 0., 3.]])
    normal = torch.tensor([[0., 0., 1.]]).expand_as(force)
    fn, ft = contract.surface_force_projection(force, normal)
    assert fn.tolist() == [5., 3.] and ft.tolist() == [0., 4.]
    assert torch.equal(force.norm(dim=-1), torch.tensor([5., 5.]))
