from __future__ import annotations

import torch
import pytest

from src.task.CmResidual.tools.analyze_crossobject_action import estimate


def test_estimate_stratifies_objects_and_scores_reset_as_zero():
    records = {
        "assignment": torch.tensor([1, -1, 1, -1, 1, -1, 1, -1]),
        "global_step": torch.tensor([50] * 4 + [60] * 4),
        "motion_id": torch.tensor([0, 0, 1, 1] * 2),
        "pre_contact": torch.ones(8, dtype=torch.bool),
        "base_action": torch.zeros(8, 18),
        "executed_action": torch.zeros(8, 18),
        "object_state": torch.zeros(8, 13),
        "followup_object_state": torch.zeros(8, 13),
        "followup_contact_count": torch.full((8,), 5),
        "progress": torch.full((8,), 10),
        "followup_progress": torch.full((8,), 15),
        "followup_reset": torch.zeros(8, dtype=torch.bool),
    }
    records["followup_object_state"][:, 2] = torch.tensor(
        [.020, .000, .040, .010, .020, .000, .040, .010])
    records["followup_reset"][3] = True
    result = estimate(records, horizon=5, num_envs=4)
    assert result["by_object"]["0"]["contact_supported_dz_plus_minus_mm"] == pytest.approx(20)
    assert result["by_object"]["1"]["contact_supported_dz_plus_minus_mm"] == pytest.approx(35)
    assert result["pooled_contact_supported_dz_plus_minus_mm"] == pytest.approx(27.5)
