import torch

from src.task.CmResidual.tools import fit_selective_causal_gate as gate


def test_policy_abstains_when_no_candidate_has_safe_lcb():
    n = 4
    # [rows, arms] point predictions and [rows, bootstrap, arms].
    supported_point = torch.tensor([
        [100.0, 100.0, 100.0],
        [103.0, 100.0, 106.0],
        [106.0, 100.0, 111.0],
        [100.0, 100.0, 100.0],
    ])
    supported_boot = supported_point[:, None, :].repeat(1, 8, 1)
    supported_boot[0, :, 0] = 104.0
    supported_boot[0, :, 2] = 104.0
    contact_point = torch.full((n, 3), 0.5)
    contact_boot = contact_point[:, None, :].repeat(1, 8, 1)
    contact_boot[1, :, 2] = 0.54  # unsafe + arm
    policy, diagnostics = gate._policy_from_predictions(
        supported_point, supported_boot, contact_point, contact_boot,
        use_uncertainty=True,
    )
    assert policy.tolist() == [0, 0, 1, 0]
    assert diagnostics["has_candidate"].tolist() == [False, False, True, False]


def test_point_gate_uses_fixed_arm_mapping():
    point = torch.tensor([[106.0, 100.0, 100.0], [100.0, 100.0, 106.0]])
    boot = point[:, None, :].repeat(1, 4, 1)
    contact = torch.full((2, 3), 0.5)
    contact_boot = contact[:, None, :].repeat(1, 4, 1)
    policy, _ = gate._policy_from_predictions(
        point, boot, contact, contact_boot, use_uncertainty=False
    )
    assert policy.tolist() == [-1, 1]


def test_ipw_value_matches_uniform_propensity_mean():
    rows = {
        "assignment": torch.tensor([-1, 0, 1, 0]),
        "held_lift": torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }
    policy = torch.tensor([0, 0, 1, 0])
    value, matched = gate._ipw_value(rows, policy, "held_lift")
    assert matched == 3
    assert abs(value - (1.0 / 3.0)) < 1e-8
