from __future__ import annotations

import numpy as np

from src.task.CmResidual.tools.analyze_randomized_followup import effects


def test_effects_use_step_strata_and_env_cluster():
    steps = np.repeat([50, 60], 4)
    assignment = np.array([1, 1, -1, -1, 1, 1, -1, -1])
    outcome = np.array([3, 3, 1, 1, 12, 12, 10, 10], dtype=float)
    result = effects({"outcome": outcome}, assignment, steps, 4,
                     bootstraps=100, seed=10)
    assert result["outcome"]["plus_minus"] == 2
