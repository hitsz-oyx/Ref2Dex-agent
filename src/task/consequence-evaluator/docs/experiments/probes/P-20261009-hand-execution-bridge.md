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
seeds: [291, 293, 294, 295, 296, 297, 298]
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

### First bridge result and bounded follow-up Decision Note

Train96/val64/test64 complete in87.84/75.57/78.24s,1440/960/960windows,
all542steps and zero clipping. Smoke8env96steps completes19.47s; initial
Torch-before-Isaac import failure is preserved and fixed before usable data.
Fit commit4615f1b,2000updates per arm,25.04s/425MiB, val selects step200both.
Held295HA/H/persistence/nominalFK all-step point RMSE is63.05/63.10/116.73/
102.71mm; HA wrist67.74mm, relative41.94mm, h24=97.27mm. Zero-A active MSE
increases only.2747%, missing the1%action gate; do not deploy this checkpoint.
Overall motion accuracy alone does not resolve action-conditioned execution.

Next Decision: distinguish input-scale suppression from insufficient usable
action signal. The uniform.5normalization makes wrist-z request.01only.02,
while standardized state channels have unit scale. Change ONLY fixed plan
units to [.01]*3+[.1]*15 in native request coordinates (no data-std
amplification); translation requests map to metres, wrist angular requests
map through pi and fingers through the native PD scale/2. Keep samples,
backbone, init, batches, optimizer, updates, teacher and val rule unchanged.
This is a method hypothesis, not a confirmed implementation bug. Retain r1.
Use one new independent ordinary64episode seed298 test before r2 fitting;
no seed search, old test tuning, fork, epoch increase, PW/C1 training or
lowering of the1%gate. Additional collection<=300s, totalcollection<=1200s;
two matched fits remain within600s aggregate. Online budget unchanged.
Failure stops automatic deployment; success authorizes the existing planned
waterfall and prospective control. Training data and old Y remain unchanged.

### Follow-up result and smaller end-to-end Decision Note

Physical-unit r2 completes2000matched updates in23.60s, same val step200.
New298test960windows: HA/H/persistence/nominalFK RMSE67.45/67.29/121.58/
102.68mm; active zero-A MSE changes-.0403%. The action gate remains FAILED;
scaling alone does not establish useful candidate sensitivity. No claim that
the bridge gate passed, and no further epochs/units search. Original r1 and
r2 test cohorts differ; their RMSE cannot be interpreted as matched gain/loss.

Root revisits the deployment stop in light of the explicit ref5 objective:
the user requires clear hand accuracy gains before connecting the chain;
both models satisfy that requirement, while the additional1%action screen
was root-imposed. It is useful to distinguish a weak bridge from an actual
prospective selector failure rather than stop solely on this aggregate proxy.
Keep the failed action gate as a limitation and change the NEXT experiment's
budget, not its result: run the common-C1 waterfall, then ONLY16full episodes
per arm/seed296and297 (32baseline/32planner), not128per arm initially.
Choose the bridge using the recorded val episode MSE only; never choose it
from old/new test or execution outcomes. Freeze that choice before inference.
This is an explicitly bounded end-to-end mechanism Probe; no deployability
or accuracy certification follows from wiring. No GT future inputs or forks.

Expansion to the already budgeted64per arm/seed requires both seeds improve
and pooled task gain>=5pp, with pooled matched-ID harm no greater than rescue.
Otherwise stop larger prospective runs and preserve the entire failed/unclear
waterfall and task evidence. At most900s per owned run, no budget increase.
The original1%gate remains failed throughout. No user or global boundary is
changed; irreversible/hardware deployment is outside this simulation scope.

The first planner296r1 fails before any candidate execution: native main
changes cwd and a relative bridge checkpoint path no longer resolves. Preserve
its FAILED manifest/log; resolve the bridge to absolute before native startup
and rerun only planner296 in a fresh r2 folder. Baseline296completed remains
valid. Audits resolve old committed source hashes via the recorded Git blob,
while data/config/weight hashes must still match immutable files exactly.

Frozen-policy planner utility is a Mission mechanism Probe, not RL training
benefit or formal Gate1. s3/full-start differs from earlier hybrid25anchorZ90.
The bridge may model conditional mean and miss contact multimodality. Nominal
FK is a fixed-target kinematic baseline; it is not an executed control rollout.
Point-distance losses alone do not certify useful candidate ranking. Full
multi-seed Validation and training-policy Cm-on/off comparison remain later.
