---
schema: ref2dex.probe.v2
probe_id: P-20261008-value-dose-calibration
experiment_id: P-20261008-value-dose-calibration
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: 98ceb76
claim_id: C3
hypothesis_family: HF-consequence-value-dose
probe_index_in_family: 1
seed_pool: probe
seeds: [261, 262, 263, 264]
decision_changed_if_positive: retain the fixed dose-calibrated split-disjoint outcome dataset for a subsequent evaluator Probe
decision_changed_if_negative: stop larger error replay sampling and record insufficient outcome support without fitting
status: PROMISING
run_id: official-value-dose-train-20261008-r1
---

# Can larger native-bounded residual doses support outcome learning?

Result: Completed192episodes, class counts41/23train,38/26val,36/28test,
11904windows/768preferences. Mechanical data coverage passes; no model fit.
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

## Completed data and audit

All three collections ran at98ceb76/GPU2,32env/two waves each. Train/val/test
elapsed122.758/121.109/119.816s, total363.683s within900s. All96assigned
interventions triggered at initial measured contact and executed full24plans;
96other episodes were nominal controls. Actor/reference/config and bank sources
passed hash guards. All owned processes exited normally and GPU2 was released.
Memory was about18.4GB; sampled utilization35..50% reflects synchronous physical
geometry export, not evaluator training throughput. No policy/optimizer/model
training occurred.

| Probe split | source seed | successes | failures | clean success | perturbed success | windows |
| --- | --- | --- | --- | --- | --- | --- |
| train | 261 | 41 | 23 | 32/32 | 9/32 | 3968 |
| val | 262 | 38 | 26 | 31/32 | 7/32 | 3968 |
| test | 263 | 36 | 28 | 30/32 | 6/32 | 3968 |

Raw sources: `outputs/consequence-evaluator/official-value-dose-{train,val,test}-20261008-r1/`
(421.28MiB total). Labeled windows:
`outputs/consequence-evaluator/official-value-dose-labeled-20261008-r1/`
(about69.4MiB). Preparation26.964s, source groups disjoint and paired ordering264.
`training_allowed=true` here means the **mechanical coverage/source gate** passes,
not semantic/manual validation or compatibility with the old local-event trainer.
New schema still needs its own success/progress/preference training adapter.

Preferences S/P/M: train132/72/52, val124/82/50, test122/83/51 (256each).
These are cross-episode training comparisons, not H-matched causal headroom.
Train gains1/2/4 yield6/2/1perturbed successes; val5/2/0, test5/1/0, with varying
sample counts recorded in `audit/audit.json`. Requested L2 across96plans
2.684..4.409; median requested/actual L2 both3.76342, median clipping fraction0.
Actual normalized commands agree with native-clipped requested controls within
5.960e-8. Clipping was measured, not assumed absent.44of77negative episodes
never achieved45continuous geometric held frames.

[Audit entry](../../../tools/audit/audit_value_outcomes.py) recomputed every
physical label, verified episode and window hashes, and produced four **train-only**
examples: nominal success, perturbed success, failed-before45hold and failure
after holding. Plot: labeled output `audit/train-weak-label-examples.png`.
This is a numerical weak-label inspection, not camera/collision-ground-truth
confirmation. Audit elapsed10.624s; plots/caches remain in owned output/tmp.

### Remaining task-label clarification before fitting

Inspection also exposes a meaningful definition boundary: current fixed S
rejects intermediate loss after the first45hold even if the object is later
regrasped and placed. The user's normal-placing decision did not explicitly
resolve recovered intermediate failures. A concise question has been sent:
should recovered grasp/normal final placing count as success, or is any loss
after stable holding a task failure? Keep existing labels unchanged pending
that clarification; do not fit a model to hide the distinction.

Diagnostic `audit/recovery-definition-diagnostic.json` tests a stricter recovery
candidate that requires a new45hold before final placing. It changes0/0/1
train/val/test negatives to positives; other apparent recoveries may not achieve
that new hold duration. This diagnostic is not a replacement rule or relabeled
dataset. Once the task criterion is clear, create a separately identified label
output if needed, preserve old provenance, then implement the independent
success/progress/preference model Probe. No official actor becomes the Mission's
final self-trained policy initialization.
