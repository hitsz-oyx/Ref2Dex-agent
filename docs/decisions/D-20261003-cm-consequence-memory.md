# Decision Memo: Cm consequence-error memory Probe

Date: 2026-10-03

## Current decision

The existing physical-value family is closed after direct selection, MVE, critic
augmentation, task representations, auxiliary targets, and recursive planning failed
to improve closed-loop policy behavior. The next high-level responsibility is a
different observation contract: use Cm as a real one-step action-consequence predictor,
then expose the **prediction error of the previous action** as a short memory of the
contact/dynamics regime at the next decision. This is not a success predictor and does
not change the task reward or success definition.

## Hypothesis

The current policy observes the resulting pose and contact proxy, but has no explicit
signal for whether its last action produced the consequence that the frozen Cm expected.
The observed-minus-Cm residual and ensemble disagreement may identify contact transitions
and model mismatch that are useful for the next action. A residual value head is the
cheapest held-only screen; a positive screen would justify one matched policy Probe with
the same residual memory appended to the actor observation.

## Probe contract

Use the existing episode-disjoint physical-value fit/held split and the frozen physical
ensemble. For every transition, compute the ensemble mean and disagreement over Cm's
object translation/velocity/contact/events/reward/terminal outputs. At the next row,
provide only the previous transition's observed-minus-predicted consequence and a valid
history mask. Fit a small residual value head on the fit rows while keeping direct-Q and
Cm frozen. A shuffled residual control uses the same architecture and budget.

The held screen passes only if the real-residual model improves direct-Q RMSE by at least
`0.5`, loses no more than `0.01` episode Spearman, has at least `0.80` valid memory rows,
and beats the shuffled-residual control by at least `0.2` RMSE. Otherwise close this
observation contract without native collection, actor changes, or scans.

## Cost and boundary

This is one held-only Probe using one available GPU, at most 120,000 fit rows and 600
updates. It consumes a new high-level budget within the existing campaign wall-time and
storage limits. It does not collect ordinary candidate data, scan thresholds/seeds/
horizons, or revive any closed HF01-HF25 route. If the screen passes, exactly one
matched policy Probe may follow; if it fails, the campaign remains frozen and the Cm
policy-utility claim remains open.

## Outcome

The full screen used `120,000` fit rows and `209,788` held rows. Valid memory coverage
was `99.82%`. Direct-Q held RMSE was `26.833`; the real residual-memory head was
`27.072`, while the shuffled control was `27.670`. Episode Spearman rose from `0.743`
to `0.769`, but the required RMSE improvement of `0.5` was not reached. Close this
observation contract and retain only the mixed episode-level clue.
