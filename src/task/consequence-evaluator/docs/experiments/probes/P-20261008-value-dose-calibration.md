---
schema: ref2dex.probe.v2
probe_id: P-20261008-value-dose-calibration
experiment_id: P-20261008-value-dose-calibration
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: 2ba0b37
claim_id: C3
hypothesis_family: HF-consequence-value-dose
probe_index_in_family: 1
seed_pool: probe
seeds: [261, 262, 263, 264]
decision_changed_if_positive: retain the fixed dose-calibrated split-disjoint outcome dataset for a subsequent evaluator Probe
decision_changed_if_negative: stop larger error replay sampling and record insufficient outcome support without fitting
status: RUNNING
run_id: official-value-dose-train-20261008-r1
---

# Can larger native-bounded residual doses support outcome learning?

Result: Pending fixed64train episode dose calibration.
Decision: Separately calibrate requested dose after0.2error replay failed
held-out class coverage; retain identical physical success labels and task.

## Decision Note

Current issue: nominal successes are plentiful; placing/holding perturbations
yield only one negative per64, and initial-contact train59/5looks usable but
independent val64/0and test63/1do not. All plans triggered and reached physics;
the initial-contact method's requested/actual L2 match. Lack of class support
blocks value learning and binary outcome evaluation, not pipeline operation.

Root action: a new bounded dose Probe, not another seed search at the old dose.
Use the same first32nominal train230observations and pinned learner/expert
projection, same-H teacher replay<=1e-5. Error gains1/2/4, clip requested residual
at0.5for rotation/finger channels; **translation cap stays0.05**; coupled distal
channels remain zero. Smooth24step chunks with zero endpoints. Actual native
controls remain clipped[-1,1]. This is simulated control dose, not hardware
operation, a new Gaussian failure law, or a change to labels/physical assets.

No external authorization boundary is crossed: current user requested continued
data generation, AGENTS permits reversible research variable changes, all work
remains in the existing repository/main and resource budget. Stop the old Probe
and preserve its results. After this new dose's gate fails, stop the recipe
rather than escalating dose or selecting successful held-out seeds indefinitely.

## Frozen sampling and pass conditions

64train at261,32parallel env/two waves, halfclean/halfinitial geometric contact.
At most one decision-known24requested plan, then the same frozen official actor
to complete reference end. No fork, force trigger, outcome feedback or recovery
controller swap. The old0.2default remains; larger mode requires an explicitly
matched bank and `--residual-bound0.5` in both collector and raw manifest.
Native requested vs clipped/executed residuals remain separate audit traces.

Train>=24complete triggers,>=24clean successes and>=4failures permits64val262
and64test263with **the same frozen bank**. No tuning using their outcomes.
All Probe seed groups; pair ordering264/stride8. Final data gate remains at least
4/2/2unique positive and negative episodes in split-disjoint train/val/test.
H/requested delta/E/I input excludes S/P/M, stage, force, bank identity and dose
category. Training preferences stay separate from H-matched headroom validation.

## Cost, stops and limitations

One idle GPU2, same-H inference<=120s; <=900s total collection,<=250s per run,
<=32env/64episodes per run, <=2GiB raw total expected. CPU labels/statistics<=120s
per step,<=1GiB windows/audit. Commit before simulation, hash sources and bank,
stop on drift/occupancy/nonfinite/incomplete data or budget. No policy/evaluator
training here, no new checkpoint/environment modification, branch or push.

Empirical error directions and larger scales are an engineering sampling choice,
not learned failure covariance. Clipping can reduce realized authority and must
be measured. Any class balancing and stage imbalance require treatment in later
fitting; enough labels do not demonstrate oracle or policy utility. Geometry and
support remain weak-label proxies pending manual formal-validation spot checks.
