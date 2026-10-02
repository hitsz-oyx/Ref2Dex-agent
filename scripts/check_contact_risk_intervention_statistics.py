"""Enumerate every possible random allocation; deliberately wrong forecasts."""
import numpy as np
from contact_risk_intervention_statistics import contributions, duplicate_contrast, MAPPING


def main():
    truth = np.arange(20, dtype=float).reshape(5, 4)/13
    mu = np.arange(20, dtype=float).reshape(1, 5, 4)/7-2
    assigned = MAPPING.copy(); outcome = truth[assigned]
    forecasts = np.repeat(mu, 16, axis=0)
    aipw = contributions(assigned, outcome, forecasts)
    np.testing.assert_allclose(aipw.mean(0), truth, atol=1e-14)
    ht = contributions(assigned, outcome, np.zeros_like(forecasts))
    np.testing.assert_allclose(ht.mean(0), truth, atol=1e-14)
    perfect = contributions(assigned, outcome, np.repeat(truth[None],16,axis=0))
    np.testing.assert_allclose(perfect, np.repeat(truth[None],16,axis=0), atol=1e-14)
    for arm,left,right in ((2,(2,3,4),(5,6,7)),(3,(8,9,10),(11,12,13))):
        contrast=duplicate_contrast(np.arange(16),outcome,forecasts,arm,left,right)
        np.testing.assert_allclose(contrast.mean(0),np.zeros(4),atol=1e-14)
    print('PASS allocation expectation, HT, perfect nuisance, duplicate-program null')


if __name__ == '__main__':
    main()
