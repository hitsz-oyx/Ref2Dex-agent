from __future__ import annotations

import numpy as np

from src.task.CmResidual.tools.probe_cm_geometry_complement import group_gap


def test_group_gap_compares_treatment_effects_within_score_groups():
    outcome = np.array([1., 1., 0., 0., 1., 1., .5, .5])
    assignment = np.array([1, 1, -1, -1, 1, 1, -1, -1])
    stratum = np.full(8, 50)
    low = np.array([True, False, True, False, True, False, True, False])
    high = ~low
    result = group_gap(outcome, assignment, stratum, low, high, np.arange(8))
    assert result["high_minus_low"] == 0
