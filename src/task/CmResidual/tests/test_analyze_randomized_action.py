import numpy as np
import pytest

from src.task.CmResidual.tools.analyze_randomized_action import weighted_step_difference


def test_weighted_step_difference_balances_within_step():
    y = np.array([1., 3., 2., 4., 10., 20., 100.])
    a = np.array([-1, 1, -1, 1, -1, 1, 0])
    s = np.array([50, 50, 50, 50, 60, 60, 60])
    difference, rows = weighted_step_difference(y, a, s)
    assert difference == pytest.approx((4 * 2 + 2 * 10) / 6)
    assert [row["selected"] for row in rows] == [4, 2]


def test_weighted_step_difference_requires_both_arms():
    with pytest.raises(ValueError, match="no randomized treatment contrast"):
        weighted_step_difference(np.array([1., 2.]), np.array([0, 1]),
                                 np.array([50, 50]))
