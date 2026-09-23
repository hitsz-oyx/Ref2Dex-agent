"""Check the paired first-grip counterfactual audit's validity gates."""
import torch

from src.task.CmResidual.tools.analyze_first_grip_counterfactual import (
    effect_metrics, match_arms,
)


def _arm(candidate: int):
    return {
        "env_id": torch.tensor([7]),
        "motion_id": torch.tensor([0]),
        "start_frame": torch.tensor([12]),
        "progress": torch.tensor([24]),
        "executed_candidate": torch.tensor([candidate]),
        "q": torch.zeros(1, 18),
        "base_action": torch.zeros(1, 18),
        "object_state": torch.zeros(1, 13),
    }


def test_exact_prestate_matches_but_mismatch_is_rejected():
    off, boosted = _arm(0), _arm(1)
    ai, bi, report = match_arms(off, boosted)
    assert ai.tolist() == bi.tolist() == [0]
    assert report["matched"] == 1
    boosted["object_state"][0, 0] = 2e-5
    ai, bi, report = match_arms(off, boosted)
    assert ai.numel() == bi.numel() == 0
    assert report["matched"] == 0


def test_effect_gate_uses_causal_difference_and_goal_direction():
    actual = torch.tensor([[0.001, 0, 0], [0.002, 0, 0]])
    predicted = actual.clone()
    goal = torch.tensor([[1., 0, 0], [1., 0, 0]])
    result = effect_metrics(actual, predicted, goal)
    assert result["pairs"] == 2
    assert result["informative_pairs"] == 2
    assert result["goal_progress_sign_agreement"] == 1.0
    assert result["effect_epe_improvement_fraction"] == 1.0
