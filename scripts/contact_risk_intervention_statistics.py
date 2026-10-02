"""Prespecified known-propensity estimates for four H10 outcomes."""
import numpy as np

PROBABILITIES = np.array([1/16, 1/16, 6/16, 6/16, 2/16])
MAPPING = np.array([0, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4])


def contributions(assignment, outcomes, forecasts):
    if forecasts.shape != (len(assignment), 5, 4) or outcomes.shape != (len(assignment), 4):
        raise ValueError('five actual arms and four frozen outcomes')
    if not np.isfinite(forecasts).all() or not np.isfinite(outcomes).all():
        raise ValueError('finite statistical inputs')
    indicator = assignment[:, None] == np.arange(5)[None]
    return forecasts + indicator[..., None]/PROBABILITIES[None, :, None]*(outcomes[:, None]-forecasts)


def duplicate_contrast(allocation, outcomes, forecasts, arm, left, right):
    weight = np.isin(allocation, left)/(len(left)/16)-np.isin(allocation, right)/(len(right)/16)
    return weight[:, None]*(outcomes-forecasts[:, arm])
