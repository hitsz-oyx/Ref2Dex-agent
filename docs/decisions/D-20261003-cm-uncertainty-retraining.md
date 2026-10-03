# Decision Memo: fixed high-uncertainty Cm retraining Probe

Date: 2026-10-03

## Question

Does reweighting Cm training toward the disagreement-defined high-error regime improve
the physical predictor enough to change its task-value usefulness?

## Contract

Use a deterministic 120,000-row fit sample. Compute the frozen ensemble disagreement on
that sample, define the high-uncertainty set as its top 20%, and fine-tune one copy of
each frozen dynamics member for exactly 600 updates using only that set. No rows from the
held split determine training, no model weights/horizon/threshold scan is performed, and
the direct-Q, actor, reward and success definition stay fixed.

The held screen reports overall and fixed high-uncertainty physical error plus the
one-step conservative direct-Q target. Pass requires top-regime physical RMSE improve by
at least 10%, overall physical RMSE not worsen by more than 2%, and one-step target RMSE
improve by at least `0.5` without episode-Spearman loss over `0.01`. A failure closes
targeted retraining and authorizes no policy run. A pass authorizes one fixed policy
Probe using the adapted physical bundle.

## Cost and boundary

One GPU, existing transitions only, at most 120,000 fit rows, 600 updates and one full
held evaluation. This is the single follow-up authorized by the HF28 acquisition signal;
it does not expand ordinary data or scan seeds, thresholds, horizons or model weights.

## Outcome

The run completed with 120,000 fit rows, 24,000 selected high-uncertainty rows, 600
updates per ensemble member and 209,788 held rows. Selected-regime physical RMSE improved
from `0.905606` to `0.859734` (`5.1%`), below the predeclared `10%` gate; overall physical
RMSE improved `3.2%` (`0.482034` to `0.466745`). The conservative value-target RMSE
worsened slightly from `26.622387` to `26.632624`, while episode Spearman changed by
`+0.00678`.

The Probe is `UNPROMISING`. Close targeted reweighting, do not run a policy follow-up, and
do not expand ordinary data or scan thresholds, seeds, horizons or model weights. The
uncertainty acquisition signal remains a data-local observation that would require actual
targeted transitions or a new decision contract before further work.
