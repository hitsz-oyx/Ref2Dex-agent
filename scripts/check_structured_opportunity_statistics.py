#!/usr/bin/env python3
"""Exact enumeration of randomized estimator expectations, including wrong mu."""
import json
import numpy as np
from structured_opportunity_statistics import MAPPING, PROBABILITIES, contributions, duplicate_contrast


def check():
    potential = np.arange(15, dtype=float).reshape(5, 3)/7 - 1
    assignment = MAPPING.copy()
    outcomes = potential[assignment]
    wrong = np.broadcast_to(np.cos(np.arange(15)).reshape(5, 3)*12, (16, 5, 3)).copy()
    value = contributions(assignment, outcomes, wrong).mean(0)
    np.testing.assert_allclose(value, potential, atol=1e-13)
    zero = np.zeros_like(wrong)
    ht = (assignment[:, None] == np.arange(5)[None])[..., None]/PROBABILITIES[None, :, None]*outcomes[:, None]
    np.testing.assert_array_equal(contributions(assignment, outcomes, zero), ht)
    perfect = np.broadcast_to(potential, wrong.shape).copy()
    np.testing.assert_array_equal(contributions(assignment, outcomes, perfect), perfect)
    for arm, left, right in ((0, (0,), (1,)), (1, (2, 3), (4, 5))):
        np.testing.assert_allclose(duplicate_contrast(np.arange(16), outcomes, wrong, arm, left, right).mean(0), 0, atol=1e-13)
    np.testing.assert_array_equal(np.bincount(MAPPING, minlength=5)/16, PROBABILITIES)
    return dict(engineering_passed=True, enumerated_allocations=16, deliberately_wrong_mu_expected_value_error=float(np.abs(value-potential).max()),
        zero_mu_equals_ht=True, perfect_mu_zero_residual=True, duplicate_program_expected_zero=True,
        scope='estimator algebra only; synthetic potential outcomes are not simulator counterfactual truth')


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
