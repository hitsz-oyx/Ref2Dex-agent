# Decision Memo: Cm two-step model-predictive planning Probe

Date: 2026-10-03

## Outcome

The held-only two-step planning screen failed. On 512 held rows, recursive MPC worsened
RMSE by `3.788` versus direct-Q and changed first-action argmaxes on `82.8%` of rows,
above the predeclared `5%–60%` range. Episode Spearman changed only `+0.009`, so the
large action changes were not supported by a useful held target improvement.

## Decision

Close this planning contract. Do not spend native simulation or policy-training budget
on it, and do not scan horizon, action panels, uncertainty thresholds, or model weights.
The result excludes recursive two-step use of the current Cm ensemble; it does not
refute Cm's one-step physical consequence prediction.

The current Cm physical-value family now has held and native evidence against direct
selection, one-step MVE, critic residual/augmentation, task representations, auxiliary
targets, and recursive two-step planning. A future continuation requires a new
observation or control responsibility with a new decision memo.
