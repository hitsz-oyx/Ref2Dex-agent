import importlib.util
from pathlib import Path
import torch

PATH = Path(__file__).resolve().parents[1] / "src/intervention.py"
SPEC = importlib.util.spec_from_file_location("intervention", PATH)
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def test_headroom_checks_every_arm_before_assignment():
    base = torch.zeros(3, 18)
    base[1, 6] = .95
    base[2, 0] = -.995
    assert contract.all_arms_have_headroom(base, contract.residuals()).tolist() == [True, False, False]
    assert contract.residuals()[:, [7, 9, 11, 13, 16, 17]].count_nonzero() == 0


def test_rotation_target_ignores_quaternion_sign_and_has_short_axis_angle():
    before = torch.zeros(2, 72)
    before[:, 6] = 1
    after = before.clone()
    after[0, 6] = -1
    after[1, 5:7] = torch.tensor([.70710678, .70710678])
    target = contract.physical_targets(before, after)
    assert torch.allclose(target[0, :12], torch.zeros(12))
    assert torch.allclose(target[1, 3:6], torch.tensor([0., 0., torch.pi/2]), atol=1e-6)


def test_drop_is_conditional_and_six_step_loss_matters():
    before = torch.zeros(3, 72)
    before[:, 2] = torch.tensor([0., .04, .04])
    before[:, 71] = 1
    trajectory = before[:, None].repeat(1, 32, 1)
    trajectory[1, :5, 71] = 0
    trajectory[2, :6, 71] = 0
    outcomes, risk = contract.window_outcomes(before, trajectory, torch.zeros(3))
    assert risk.tolist() == [False, True, True]
    assert outcomes[:, 3].tolist() == [0., 0., 1.]
    assert outcomes[0, 2] == 0
    assert outcomes[1, 2] == 11/16
