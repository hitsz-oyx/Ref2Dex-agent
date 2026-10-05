---
schema: ref2dex.probe.v2
probe_id: P-20261006-y-noise-tolerance
experiment_id: P-20261006-y-noise-tolerance
date: 2026-10-06
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: bd111d34e77b8e61e624fc375e1820ea1e22682a
claim_id: C3
hypothesis_family: HF-y-noise-tolerance
probe_index_in_family: 1
seed_pool: probe
seeds: [100, 101, 102, 103, 104]
decision_changed_if_positive: set a minimum offline ranking target before fitting a Y predictor
decision_changed_if_negative: retain the rolling GT-Y result but do not invest in the current Y predictor contract
status: PROMISING
---

# Gate 3: offline short-Y ranking/noise tolerance

## Decision Note

The completed ref14_1 rolling oracle is PROMISING (27/32 versus 23/32 baseline,
4 rescued, 0 harmed). Before fitting a predictor, this Probe asks the cheapest
next question: how much ranking degradation can the fixed short-Y utility tolerate
on saved same-current-state candidate panels? A positive result defines a minimum
ranking target; a negative result keeps the oracle evidence but stops this fixed
predictor contract. This is an offline Decision Probe, not a closed-loop control
experiment.

## Frozen protocol

Read only `outputs/cm-interaction-oracle/rolling-oracle-control-s263-s264/*-plan.json`.
Use only plans with `baseline_upper_bound_certificate=false`, so all seven
candidate Y panels are physically observed. A panel row is one same-current-state
query; do not combine rows across query offsets or stitch trajectories. Preserve
the existing eight Y channels and utility `U=Y7+0.25*Y3-Y6`; do not retune
thresholds or labels. Add independent zero-mean Gaussian noise to all eight
channels with scalar sigma in `{0, .02, .05, .10, .20}` and seeds `[0,1,2,3,4]`.
The sigma grid is a diagnostic unit-scale grid, not a claim about a predictor's
calibration. Report strict-pair accuracy (prediction ties receive 0.5), GT-tied
pair count, top-1 regret and panel coverage. Use the saved GT utility as truth.

The output cannot report oracle-gain retention or policy Z: noisy scores are never
executed. No new PhysX, model fitting, action execution or Y representation change
is allowed. Results are descriptive over repeated rows from four shared-solver
groups; they are not a Validation or an independent environment bootstrap.

## Stopping and interpretation

The Probe is adequate if at least 100 fully observed same-state panels are read.
`PROMISING` requires a nonzero-noise row with median pairwise accuracy at least
0.70 and median top-1 regret no greater than 0.125 utility units; otherwise the
fixed predictor target is `UNCLEAR` or `UNPROMISING` as appropriate. These are
offline screening thresholds only. Any learned predictor must later pass the same
ranking contract and a fresh real rolling intervention before claims about Z.

## Provenance and limits

The source rolling result is the completed ref14_1 artifact and its immutable plan
hashes. Missing candidates are excluded rather than imputed. The exposed cohort,
shared solver groups, binary Y channels, and scalar noise units limit interpretation.

## Result

The completed offline audit (`outputs/cm-interaction-oracle/y-noise-tolerance-20261006-v2/result.json`)
used 43 fully observed plans and 339 same-state panels
(all seven candidate Y vectors present). Median pairwise accuracy was 0.931 at
sigma .02, 0.840 at .05, 0.758 at .10, and 0.696 at .20; the corresponding
median top-1 regret was 0.0 utility units for this tie-heavy panel set. The frozen
screening gate is `PROMISING` because nonzero noise through sigma .10 remains above
0.70 pairwise accuracy with median regret at most .125. Sigma .20 falls below the
pairwise target. These numbers are offline ranking diagnostics only and do not
estimate real rolling Z retention.
