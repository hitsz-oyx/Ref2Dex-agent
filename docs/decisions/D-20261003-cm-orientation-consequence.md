# Decision Memo: Cm object-orientation consequence Probe

Date: 2026-10-03

## Question

Does the current conservative value interface discard useful Cm information by keeping
the observed object orientation in the one-step continuation? The physical model predicts
an action-conditioned object quaternion, but all recent MVE and model-based critic paths
used the observed orientation to avoid model error.

## Contract

Run one held-only screen on the existing episode-disjoint physical-value split. Cm keeps
the observed hand q/dq and supplies predicted object translation/velocity/contact/events;
the treatment additionally supplies the frozen ensemble's predicted object orientation
to the direct-Q continuation. No actor observation, reward, success predictor, model
weight, uncertainty gate, horizon, or action panel changes.

The treatment must improve frozen one-step target RMSE by at least `0.5`, lose no more
than `0.01` episode Spearman, and have finite orientation errors. A failed screen closes
this orientation contract with no critic fine-tune or policy Probe. A passing screen
would authorize exactly one fixed critic fine-tune and then a matched policy Probe.

## Cost and boundary

This is a single held-only forward screen using the frozen ensemble and existing data,
one GPU, and no new collection. It is a new consequence representation, not a scan of
the closed physical-value family. It does not revive HF01-HF25 or change the mission
claim; the allocated HF27 budget is one Probe.

## Outcome

The observed-orientation control had held RMSE `26.622`; supplying the Cm-predicted
orientation gave `26.642`, with episode Spearman changing `+0.0006`. The quaternion
error was `1.35 rad` mean and `2.74 rad` at P90. The RMSE gate failed, so close this
contract without critic fine-tuning or policy training.
