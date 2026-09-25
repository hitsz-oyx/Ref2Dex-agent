# P-20260925-cm-option-value-retest

- Classification: Decision Probe; one untuned repeat after a near-threshold
  action-information signal.
- Model, experts, motions and train data: identical to
  `P-20260925-cm-option-value`.

## Question and decision

The seed224 action-aware Brier was 9.66% below blind but the fixed
router still scored one more grasp. Is this action information stable
on one more unseen start distribution? Refit the exact predeclared
training procedure using **only seed223**, including PCA, scaling,
regularization and expert order. Hold out all of new seed225 for one
evaluation. Use the same five frozen experts, 64 first full episodes
each, and one actual fixed-route rollout. For the shuffled control use
the unchanged train RNG seed20260925223 and new independent test RNG
seed20260925225. Do not tune model features or gates on seed224/225.

Continue to a small online Cm routing Probe only if action-aware Brier
is >=10% lower than blind and >=5% lower than shuffled, and offline
top-one expert selection exceeds both blind and the actual fixed route
by >=5/64 on seed225. Otherwise stop this initial-action route and
investigate action choices at contact, where effects are less diluted
by the full episode. This remains exploratory; a passing offline
selector would still need an executed matched Cm-on/off comparison.

One idle GPU, <=20 minutes and <300 MB outputs. Stop on data drift,
state misalignment, incomplete episodes or GPU conflict.

## Results

Pending.
