import importlib.util
from pathlib import Path
import torch
import numpy as np

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


def test_early_hold_run_needs_lift_and_contact_and_resets_on_loss():
    previous = torch.tensor([5, 5, 5, 0])
    height = torch.tensor([.031, .029, .05, .04])
    contact = torch.tensor([True, True, False, True])
    result = contract.update_predecision_hold(previous, height, torch.zeros(4), contact)
    assert result.tolist() == [6, 0, 0, 1]


def test_continuation_keeps_early_failed_trials_and_lost_run_crosses_step8():
    before = torch.zeros(3, 72)
    before[:, 2] = .04
    before[:, 71] = 1
    trajectory = before[:, None].repeat(1, 32, 1)
    trajectory[0, :, 2] = .01
    trajectory[1, 4:10, 71] = 0
    trajectory[2, :8, 71] = 0
    outcomes, details = contract.continuation_outcomes(before, trajectory, torch.zeros(3))
    assert details["first_failure_step"].tolist() == [1, 10, 6]
    assert outcomes[:, 5].tolist() == [1., 1., 0.]
    assert details["all32_failure"].tolist() == [True, True, True]
    assert outcomes[2, 3] == 1
    assert details["late_height_failure"].tolist() == [True, False, False]


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


def test_repeated_environment_never_crosses_holdout_boundary():
    path = Path(__file__).resolve().parents[1] / "tools/run/probe_interventions.py"
    spec = importlib.util.spec_from_file_location("intervention_probe", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    clusters = np.tile(np.arange(30), 4)
    motion = clusters % 3
    train, test = probe.split_stratified(motion, clusters, 211)
    assert len(test) == 24
    assert not set(clusters[train]) & set(clusters[test])
    assert sorted(np.concatenate((train, test)).tolist()) == list(range(120))


def test_drop_ranking_excludes_prelift_and_score_tie_gets_half_credit():
    path = Path(__file__).resolve().parents[1] / "tools/run/probe_interventions.py"
    spec = importlib.util.spec_from_file_location("intervention_probe_ranking", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    target = np.zeros((2, 8))
    target[1] = 1
    prediction = np.zeros_like(target)
    result = probe.ranking(prediction, target, np.arange(2), np.zeros(2, dtype=int), np.arange(2))
    assert result["macro"] == .5
    assert result["per_head"][3] is None
    assert result["support"][3]["pairs"] == 0


def test_early_hold_primary_rank_uses_retention_and_failure_after_step8():
    path = Path(__file__).resolve().parents[1] / "tools/run/probe_early_hold.py"
    spec = importlib.util.spec_from_file_location("early_hold_probe_ranking", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    target = np.zeros((2, 6))
    target[1] = 1
    groups, arms, ids = np.zeros(2, dtype=int), np.arange(2), np.arange(2)
    scores = target.copy()
    scores[:, 5] *= -1
    result = probe.retention_ranking(scores, target, ids, groups, arms)
    assert result["primary"] == .5
    assert result["per_head"][3] == 1
    assert result["per_head"][5] == 0
