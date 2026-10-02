# Decision Memo: Cm critic residual interface

Date: 2026-10-03

## Question

Do frozen Cm physical consequences add reliable held task-value information beyond
the direct-Q baseline when used only in a critic residual head?

## Evidence

The fixed physical checkpoint and fit/held split contained 831,811 fit and 209,788
held transitions. Direct-Q scored held RMSE `26.833`, row Spearman `0.589`, and
episode Spearman `0.743`. A current-state/action residual worsened the held score.
The Cm residual reduced neither row error nor row ranking: RMSE `27.050`, MAE
`9.589`, and row Spearman `0.586`. It raised episode Spearman to `0.761`, a small
`+0.018` change without a consistent metric improvement.

## Decision

Close this exact linear critic-residual interface as `UNCLEAR`. The episode-level
increase is retained as a weak critic-only clue, but it is insufficient to justify
policy training and does not override the negative native action-utility screens.
Do not scan Ridge settings or launch PPO. A future attempt must change the value
representation or decision target and define its held criterion before execution.

## Cost and safety

The audit used the frozen checkpoint and existing fit/held collections on one idle
GPU. It wrote a new result file only; no checkpoint, external project, or running
process was modified.
