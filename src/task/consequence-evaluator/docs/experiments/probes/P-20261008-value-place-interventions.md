---
schema: ref2dex.probe.v2
probe_id: P-20261008-value-place-interventions
experiment_id: P-20261008-value-place-interventions
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: 8245f53
claim_id: C3
hypothesis_family: HF-consequence-value-interventions
probe_index_in_family: 1
seed_pool: probe
seeds: [231, 232, 233, 245]
decision_changed_if_positive: freeze the bounded train-derived sampler and prepare independent value training and held-out outcome windows
decision_changed_if_negative: preserve failed sampling evidence and inspect intervention authority or physical labels before expanding collection
status: UNPROMISING
run_id: official-value-place-train-20261008-r1
---

# Can empirical placing deviations supply informative outcome supervision?

Result: Completed64episodes; clean32/32success, placing31/32success. All32plans
triggered at482and executed completely; only one train negative. Coverage gate failed.
Decision: The user authorized continuing targeted data collection. Keep the
full-reference grasp/lift/controlled-place task and fixed geometric label rule.
No evaluator training or change to the Mission's final policy claim here.

## Motivation and Decision Note

The [nominal pilot](P-20261008-full-reference-value-data.md) produced62success
and2failure episodes at seed230. Both failures achieved stable holding and
lost support during final placing (first geometric onset504/507). A uniformly
noisy entire episode would spend budget outside the observed weak stage.

This Decision Probe asks whether bounded empirical pre-failure control-error
candidates actually execute and yield both positive and negative complete
outcomes, retaining clean controls. A positive result justifies preparing a
small independent train/val/test value dataset; a negative result stops expansion
and directs the next decision to action authority/label diagnosis. The cheapest
method is64episodes,32parallel environments/two waves, half nominal/half placing
intervention at train seed231. At least24full actual placing triggers, at least
24clean successes and at least4train failures are required before further groups.

## Frozen sampler and clocks

Build bank only from the completed nominal **train seed230** source and its
already frozen S/P/M labels. Remove recorded actual-residual execution noise
from normalized controls to recover the policy base. For each negative episode,
take up to24controls before its first post-reference-placing unheld/unsupported
state, starting no earlier than reference placing. Compare those controls with
the same-tick median of62positive nominal episodes. Smooth with four temporal
median bins/interpolation, sin-squared taper and zero endpoints. Three gains
(0.5/1/1.5) produce6candidates; clamp absolute residual to0.2 and wrist translation
to0.05; zero the six Inspire-overwritten distal channels. Uniformly select a
candidate once per episode; record its member/gain/source identity.

This is an **empirical candidate bank from two failures**, not a reliable fitted
failure distribution, CARE reproduction, DART learner-error covariance, or a
claim that the observed policy deviations caused the original failures.

For value collection, phases use measured hand-object surface proximity,
unsupported3cm elevation and45continuous held frames. Placing may trigger only
after a previous45frame hold, current near-hand geometry and the fixed full
reference's final placing boundary (482). Native net force is diagnostic only.
Clean/placing allocation is randomized within motion and balanced across waves.
Each episode contains at most one known24step chunk and then the same frozen
official feedback controller to the complete first episode end; no fork/reset.
Pre-trigger intervention windows are unknown and excluded, rather than promising
a future intervention whose state trigger is not yet known.

If the first train batch passes the stated coverage gates, use the frozen bank
without looking at held-out outcomes to collect64val episodes at232 and64test
episodes at233 (same clean ratio/controller/reference). These are **Probe** seed
groups, not formal validation holdout. Prepare all three sources using pair
seed245/stride8 and the unchanged S/P/M physical weak-label parameters. Minimum
success/failure episode counts4/2/2 still apply; readiness is not permission to
use the old local-preference trainer, which rejects this independent schema.

## Resources, provenance and stops

One idle GPU2 and at most900s total collection (each run hard cap250s), with
32environments; <=2GiB total expected new raw data (per-run existing bound2GiB).
CPU bank/label/statistical processing<=120s each and <=1GiB total windows/audit.
Existing official SHA8f6823db…, configs, motion and complete archived Torch2.0.1
runtime are unchanged. Code is committed before collection; source/bank/file
hashes are verified before and after each run. Stop on foreign GPU occupancy,
source drift, nonfinite/missing diagnostics, partial chunk or exhausted budget.
No environment/checkpoint modification, external writes, new branch or push.

## Limitations and future evidence

R1 completed at8245f53/GPU2 in117.478s. Raw output:
`outputs/consequence-evaluator/official-value-place-train-20261008-r1/`.
Label preparation took7.081s;2240windows (clean65each, placing5each),
256preferences (6success/164progress/86margin). Labeled output:
`outputs/consequence-evaluator/official-value-place-train-labeled-20261008-r1/`,
`training_allowed=false`. No held-out expansion or fit was started.

Intervention audit found0/32already-supported at trigger; requested peak
0.0526..0.1726, median chunk L2=0.48257. Logged actual delta matched requested
plans exactly for this run; median projection1.0. Thus this is not an absent
trigger, overwritten distal-channel or clipping implementation failure.
The single negative ended settled on the table but had unsafe unsupported
hand loss during placing, so settling alone did not falsely promote S to1.
The same-reference, bounded placing-error sampler is UNPROMISING for producing
the required negative coverage; this does not refute the evaluator/Cm route.

Decision: preserve R1 and stop val/test expansion. Next Probe estimates actual
same-H learner/expert action error on nominal train holding observations and
uses a separate frozen bounded holding bank. This is stage-conditioned
DART-inspired empirical replay, not a claimed fitted CARE failure law or full
DART algorithm. Previous source hashes/labels remain historical and unchanged.

Only one object/reference and a two-negative bank. This cannot establish optimal
Q, generalization, action-causal utility, or a trained evaluator's oracle benefit.
Geometry support remains a sampled tabletop proxy; manual video/collision spot
checks are deferred until a proposed formal comparison. Exported task labels
and trigger/bank identities are sidecars, excluded from H/delta/E/I input.
Train-derived sampler reuse across splits is intentional; source episodes and
seed groups are disjoint. Reusing the previous nominal windows is unnecessary:
they retain their own historical source hashes and labels unchanged.
