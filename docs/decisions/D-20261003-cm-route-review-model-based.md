# Decision Memo: Cm route review after model-based critic Probe

Date: 2026-10-03

## Question

Does the offline model-based critic improvement justify another Cm policy integration
attempt, or should the current physical-value decision family be paused?

## Evidence

The conservative synthetic-transition Probe improved held direct-Q metrics after a fixed
critic fine-tune: RMSE `26.833 -> 25.712`, MAE `9.108 -> 8.377`, row Spearman
`0.589 -> 0.606`, and episode Spearman `0.743 -> 0.774`. This establishes a useful
value-equivalence clue under the offline transition contract.

The matched policy Probe used the same source, reward, training seed, horizon, and
evaluation contract. On the first complete evaluation seed, direct-Q reached `10/96`
stable successes and the model-based-Q arm reached `4/96`. The predeclared per-seed
nonnegative gate failed, so the second treatment seed was stopped. Earlier selector,
MVE, nonlinear representation, and auxiliary-target policy Probes also failed their
policy gates. The model-based Q changed candidate argmaxes in an offline panel, so the
failure is not evidence that Cm was never consulted; it is evidence that this critic
conversion did not improve closed-loop policy behavior.

## Decision

Close the current Cm physical-value decision family: no more Q-target weights,
uncertainty thresholds, action-teacher variants, ordinary candidate data, PPO runs, or
seed/epoch scans. Keep the physical consequence model and the offline critic result as
research evidence. The matched Cm policy-utility claim remains open and unsupported;
the route would need a genuinely new observation or planning contract before another
Probe is justified.

## Cost and safety

All work used existing read-only source data and the workspace. Own training/evaluation
processes were stopped after the predeclared early gate; no external process or data was
modified.
