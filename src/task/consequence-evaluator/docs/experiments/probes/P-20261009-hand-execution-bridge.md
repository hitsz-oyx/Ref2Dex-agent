---
schema: ref2dex.probe.v2
probe_id: P-20261009-hand-execution-bridge
experiment_id: P-20261009-hand-execution-bridge
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: 111c2b5
claim_id: C3
hypothesis_family: HF-hand-execution-bridge
probe_index_in_family: 1
seed_pool: probe
seeds: [291, 293, 294, 295, 296, 297]
decision_changed_if_positive: connect the frozen bridge to PW and C1 for prospective rolling task control
decision_changed_if_negative: isolate common wrist or candidate sensitivity failures before spending online control budget
status: UNCLEAR
run_id: hand-execution-20261009-r1
---

# Can ordinary e260 interventions bridge plans to useful future hand geometry?

Result: Pending bounded Probe requested by user ref5.
Decision: Stop expanding fork panels and observed-hand-only C2a. Freeze old
U32, actor e260, PointWorld best46000 and C1 step1000. Learn only a prospective
hand execution bridge from ordinary random intervention rollouts.

## Motivation and Decision Note

Mission C3 needs decision-known action-conditioned consequences and actual
task utility. The remaining interface blocker is H+A -> future hand11x3x24.
The least expensive discriminating experiment uses one random plan per actual
trajectory window, not seven physical futures. No new fork or GT-hand deployment.

Use airplane s3 only, full reference frame0, fixed self-trained recovered e260
SHA8882fabd, native GPU PhysX/CPU tensor pipeline as inherited compatible runtime.
Actor, bridge, PW and evaluator neural computations all use GPU2. CPU tensors
are the preserved simulator interface, not a switch to CPU model fitting.
Dense surface-gap diagnostics also use GPU, in bounded16env chunks.
Explicit process-local reset compatibility and native reference alignment apply
identically to collection and evaluation. Native hybrid probability1 selects
frame0; require start_times=0. One immutable checkpoint/config/motion bundle.

Random collection: train96episodes seed293, val64 seed294, test64 seed295.
At tick40 then every32steps with at least32steps remaining, sample uniform
old seven candidates; first8 residual steps followed by16zero. No amplitude
jitter, full feedback actor suffix. Known requests are separate from clipping
and executed controls. Whole seed/episode splitting; no future action, force,
outcome or future object/root enters bridge inputs. Capture4past geometry
states plus raw H1442,24known residuals and measured24hand point targets.

Bridge: shared MLP, separate common wrist translation and wrist-relative
finger-point change heads, composed into24x11x3 in fixed current-object frame.
Train-only H/geometry normalization, fixed plan units.5, fixed output units1cm.
Single seed291, <=2000updates AdamW3e-4, batch128, val episode point-MSE selects
checkpoint. Evaluate exactly once on seed295 after freezing. Hand persistence
and zero-motion are the same reference; disclose their equivalence. Nominal FK
holds current actor action and current q/root fixed, applies each known residual
through exact native PD conversion; never uses future feedback or measured root.

Exploratory bridge gate: test all-horizon point RMSE at least10% below both
persistence and nominal FK, h24 RMSE below persistence, finite predictions,
and test nonzero-plan MSE increases >=1% after replacing A by zero. Report
per-candidate, wrist and relative error, within-episode uncertainty. This gate
is a screening rule, not a formal causal or generalization claim. If it fails,
audit implementation/coverage and stop deployment rather than extend epochs
or fit against test labels. One predeclared H-only matched ablation may isolate
whether action information is learned; same budget/samples/initialization.

If gate passes, use same frozen C1 throughout GT hand+GT object, GT hand+PW
object and predicted hand+PW object waterfall on ordinary held decisions (not
candidate panels); summarize score error/teacher RMSE, not same-H pair ranking.
Then a prospective7candidate planner selects every8steps from current measured
H and4past geometry, G -> PW -> C1; execute only first8 steps, observe and replan.
No measured future trajectory, physics rollout, outcome or contact gate enters
selection. Full-reference baseline vs planner matched seeds296/297,64episodes
per arm/seed (128each), initial states and env layout frozen; all episodes reported.
No seed or outcome selection. Final metrics task success, stable lift45frames,
post-lift loss/recovery and average nonzero interventions. Old U is only teacher.
Use existing support/gap/settling thresholds; permit recovered success after
reestablishing stable45-frame hold before controlled placement. Bootstrap is
descriptive; shared-world and two exploratory seed clusters limit conclusions.

## Resources and stops

All new artifacts under outputs/consequence-evaluator/hand-execution-*.
One idle GPU2. Smoke<=120s; collection<=300s/run,<=900s total; bridge fit and
ablation<=600s combined; waterfall<=180s; prospective execution<=900s per arm,
<=3600s total. Entire first Probe<=5400s and<=4GiB new artifacts. Monitor GPU
memory/utilization and ETA, stop on foreign occupancy, nonfinite values, input
hash drift, missing24target coverage, reset/command mismatch, or deadline.
No actor/C1/PW fitting, new branch, remote push, external writes, overwrite of
existing runs, or change to old Y. Save code before every actual execution phase.

## Limitations / future evidence

Frozen-policy planner utility is a Mission mechanism Probe, not RL training
benefit or formal Gate1. s3/full-start differs from earlier hybrid25anchorZ90.
The bridge may model conditional mean and miss contact multimodality. Nominal
FK is a fixed-target kinematic baseline; it is not an executed control rollout.
Point-distance losses alone do not certify useful candidate ranking. Full
multi-seed Validation and training-policy Cm-on/off comparison remain later.
