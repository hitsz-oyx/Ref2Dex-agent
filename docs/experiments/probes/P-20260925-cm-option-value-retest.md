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

All six seed225 runs completed and aligned with the seed223 training
states by environment, motion and start frame. The exact seed223-fitted
procedure reproduced the earlier seed224 report byte-for-byte before
the new collection. New heldout seed225 contained 61 held-lifts among
320 expert-state rows. Results:

| Model | Heldout Brier | Offline selected route |
| --- | ---: | ---: |
| Action-aware | 0.15445 | 20/64 |
| Action-blind | 0.16229 | 11/64 |
| Action-shuffled | 0.17252 | 14/64 |
| Fixed object route | — | 20/64 actual rollout |

Action-aware Brier improved 4.83% versus blind and 10.47% versus
shuffled. It improved offline option choice over blind but **tied** the
fixed route. The predeclared action-information and route-utility gate
again **failed**. Across the two untouched heldout seeds, aware
offline selection was 40/128 and the actual fixed route was 41/128;
neither seed showed a grasp gain over fixed. Result: `UNPROMISING` for
using the initial 18-D action in this episode-outcome Cm to replace
the existing router. This is not a general refutation of Cm or of
contact-stage physical modeling. Stop this specific start-of-episode
selector and test decision points after contact, where the action has
a shorter causal horizon to the measured effect.

Artifacts: `outputs/CmResidual/agent_cm_option_value_retest_s225/` and
`outputs/CmResidual/agent_cm_option_value_retest_probe_20260925/report.json`.
