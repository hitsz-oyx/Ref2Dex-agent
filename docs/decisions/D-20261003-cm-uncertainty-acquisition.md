# Decision Memo: Cm uncertainty-guided data responsibility Probe

Date: 2026-10-03

## Question

Can Cm's ensemble disagreement identify where more action-consequence data would be
useful, so that data collection rather than another value interface becomes the next
policy-learning mechanism?

## Contract

On the fixed held physical-value split, compute disagreement over the frozen ensemble's
object translation/velocity/contact/reward/terminal predictions for the actual action.
Compare it with the realized normalized physical prediction error and with the absolute
direct-Q return residual. No target is used to fit Cm, no action is changed, and the
orientation head is excluded because its separate Probe failed.

The screen is informative only if the top-disagreement 20% has at least `1.25x` the
physical error of the bottom 50%, contains at least `30%` of absolute direct-Q residual
mass, and spans at least `100` held episodes. Otherwise close uncertainty-guided data
acquisition without collecting or training. Passing would authorize one targeted
acquisition/retraining Probe with a fixed high-disagreement sampling rule.

## Cost and boundary

One held-only forward pass over existing data, one GPU, no new simulator collection and
no policy changes. This is a new data responsibility (HF28), not a threshold, seed,
horizon, candidate-panel or ordinary-data scan.

## Outcome

The top-disagreement 20% had `7.03x` the physical error of the bottom 50%, carried
`32.5%` of absolute direct-Q residual mass, and covered all `384` held episodes. The
three gates passed, so authorize one fixed high-uncertainty fit/retraining Probe. This
is a data-allocation signal only; it is not a Cm policy-utility result.
