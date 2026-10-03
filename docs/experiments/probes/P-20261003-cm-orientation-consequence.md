# P-20261003-cm-orientation-consequence

Family: Cm orientation consequence
Type: Held-only decision Probe
Status: COMPLETED — `UNPROMISING`

## Question

Does the one-step continuation lose useful action information by replacing Cm's
predicted object orientation with the observed orientation?

## Contract

On the fixed 209,788-row held split, the frozen ensemble predicted object
translation/velocity/contact/events and orientation. Hand q/dq remained observed. The
control supplied observed object orientation to the direct-Q continuation, matching the
recent conservative MVE/critic paths. The treatment supplied the Cm-predicted canonical
quaternion. Actor, reward, success definition, model weights, horizon and action panel
were unchanged.

## Result

Direct-Q held RMSE/MAE/row Spearman/episode Spearman were
`26.833/9.108/0.589/0.743`. The observed-orientation control was
`26.622/8.873/0.590/0.740`; the Cm-predicted-orientation treatment was
`26.642/8.895/0.590/0.741`. Treatment versus control changed RMSE by `+0.019` and
episode Spearman by `+0.0006`, failing the predeclared `0.5` RMSE improvement gate.
Quaternion error was large (mean `1.35 rad`, P90 `2.74 rad`).

Close this orientation contract. Do not fine-tune the critic, collect native data, or
train a policy from it. This result does not refute Cm's translation/contact prediction.

Evidence: [results](P-20261003-cm-orientation-consequence-results.json), [Decision Memo](../../decisions/D-20261003-cm-orientation-consequence.md).
