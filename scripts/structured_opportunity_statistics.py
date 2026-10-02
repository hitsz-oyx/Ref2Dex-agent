"""Fixed pre-allocation AIPW contributions for randomized H10 outcomes."""
import numpy as np

PROBABILITIES = np.array([.125, .25, .25, .1875, .1875])
MAPPING = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])


def contributions(assignment, outcomes, forecasts):
    """[row, arm, metric]; forecasts are fixed ensemble means, not outcomes."""
    if forecasts.shape != (len(assignment), 5, 3) or outcomes.shape != (len(assignment), 3):
        raise ValueError('five randomized arms and three prespecified outcomes')
    if not np.isfinite(forecasts).all() or not np.isfinite(outcomes).all():
        raise ValueError('nonfinite statistical input')
    indicator = assignment[:, None] == np.arange(5)[None]
    return forecasts + indicator[..., None]/PROBABILITIES[None, :, None]*(outcomes[:, None]-forecasts)


def duplicate_contrast(allocation, outcomes, forecasts, arm, left, right):
    """Duplicate programs share the same augmentation, which cancels exactly."""
    weight = np.isin(allocation, left)/(len(left)/16) - np.isin(allocation, right)/(len(right)/16)
    return weight[:, None]*(outcomes-forecasts[:, arm])
