from __future__ import annotations

import numpy as np

from src.task.CmResidual.tools.analyze_multiaxis_followup import axis_effect


def test_axis_effect_uses_only_its_randomized_treatment_and_strata():
    assignment = np.tile([1, -1, 2, -2, 3, -3], 8)
    steps = np.repeat([50, 60], 24)
    outcome = np.zeros((48, 3), dtype=float)
    outcome[:, 0] = np.where(assignment == 1, 7,
                             np.where(assignment == -1, 2, 100))
    outcome[24:, 0] += 1000  # must cancel through step stratification
    part = {"assignment": assignment, "steps": steps,
            "num_envs": 24, "values": {"h10_object_mm": outcome}}
    result = axis_effect([part], 0, "h10_object_mm", bootstraps=100, seed=1)
    assert result["plus_minus"] == 5
    assert result["effective_bootstraps"] >= 90
