---
schema: ref2dex.probe.v2
probe_id: P-20261011-competent-joint-bc
experiment_id: P-20261011-competent-joint-bc
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 3093877
claim_id: C3
hypothesis_family: HF-trajectory-policy-competent-joint-bc
probe_index_in_family: 1
seed_pool: probe
seeds: [183, 184]
decision_changed_if_positive: retain autonomous H/G trajectory BC baseline before PointWorld integration
decision_changed_if_negative: diagnose earliest high-level error against competent frozen R before RL or Cm
status: UNPROMISING
run_id: competent-joint-bc-fit-20261011-r1
---

# Competent teacher H/G -> trajectory -> frozen actual-PD R

## Motivation / decision

ref3 first requires a physically competent autonomous trajectory baseline. The
[actual-PD R physical Probe](P-20261011-competent-pd-physical.md) passed 8/8
hold45 and controlled placement with independent actual-input/control audit.
Does a measured-history actor preserve that competence? This Decision changes
whether to advance to PointWorld/RL or diagnose high-level learning first.
The cheapest discriminator is one matched offline fit and one frozen physical
comparison; no hyperparameter sweep, clock input, oracle action substitution or
claim of Cm/RL improvement.

## Frozen design

Reuse audited official teacher r2, whole-env train0:16/val16:24/test24:32.
H328 contains four past/current measured states, padded with frame0 at startup;
G12 is known terminal-reference object target relative to the current object.
No future measured object/q, teacher actor features, phase, clock or force.
Independent MLP340->256Tanh->256Tanh->24x33 actual hand displacement in current
object frame. Train-only statistics, minimum std1mm, zero final layer.
Freeze audited tau R (SHA414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f).
Matched initialization/batches: tau_only uses normalized trajectory MSE;
joint uses that MSE +0.1*normalized actual-PD MSE through differentiable frozen R.
Seed183, Adam3e-4, batch512,3000updates; batch quarters startup ticks<8 and
8:64, remaining half all ticks. Temporal sampling is offline only. Select each
model's lowest validation total objective every250updates; test never selects.
Record PD gradients into actor and immutable R state. Offline execution screen:
finite losses/outputs, startup<8 first8 hand-coordinate RMS<=20mm, clipping<1%.
This screen decides bounded physics expenditure, not task competence.

## Frozen physical comparison

Seed184,24 native GPU-pipeline envs,542controls:teacher4,GT-tau/R4,
tau_only8,joint8. GT donors24:28. Each independent actor queries every8controls
from its own measured H/G, emits24 future hand points, transforms them to world
using the query's hand/object frame. Every low tick R reconditions remaining
world plan on its own current measured state; pad tail with last plan point.
The autonomous arms receive no source future hand or PD. No model selection
or refit on execution/test outcomes. Record query inputs/plans, per-control
R inputs, predictions and requested/applied actualPD for independent replay.
Calibration: teacher and GT-R each>=3/4 hold45+controlled-placement successes.
Joint gate:>=6/8 hold45+controlled placement, median held>=80% GT-R,
clipping<1%. All gates->PROMISING; valid calibration and joint fail->UNPROMISING;
failed calibration->UNCLEAR. Report tau_only separately: both successful means
usable baseline, not proof PD loss helps; require joint>=2/8 advantage for a
local incremental signal. Exact target pose success is a separate absent gate.

## Resources / stops

One idle GPU0, fit<=300s, physics<=300s, audit<=120s,<=256MiB outputs; all costs
share the original competent-PD route's20GPUmin/1GiB envelope (not extra budget).
Stop on nonfinite, data/controller/R drift, foreignGPU process, reset or timeout.
No Campaign change, no new branch, no push, no external modification.

## Results

Fit `b00a533`, seed183,3000updates/22.97s. Validation startup first8 coordinate
RMS tau_only0.922mm,joint0.693mm; normalized PD MSE .040412/.028681 (29% lower),
no offline clipping. Both selected update3000, both execution screens pass.
Independent fit audit exact predictions; normalization max2.39e-7; PD gradient
into actor18.9866, frozen R unchanged. Task tests33 passed.

Completed physical r3 `6a75a26`,seed184,542controls/65s:teacher4/4 and GT-R4/4
hold45+controlled placement, median440.5/436.5 frames; tau_only0/8,joint0/8,
median held0/0, clip4.3589%/8.8561%. Valid calibration with failed joint gate:
UNPROMISING for these fixed checkpoints with eight-control replanning.
Independent audit `4473013` exact H/G and actor predictions; plan2.38e-7m,
live-R tau2.98e-7m, R7.15e-7,requestedPD8.34e-7,actual nativePD exact0;
outcomes and frozen gates reproduce. Neither high-level arm lifts the object.

Artifacts: [fit](../../../../../../outputs/trajectory-policy/competent-joint-bc-fit-20261011-r1/result.json),
[physical](../../../../../../outputs/trajectory-policy/competent-joint-bc-execution-20261011-r3/result.json),
[audit](../../../../../../outputs/trajectory-policy/competent-joint-bc-execution-20261011-r3/audit.json),
[behavior](../../../../../../outputs/trajectory-policy/competent-joint-bc-execution-20261011-r3/behavior.png).
First actions are close to calibration, but approach deviates before contact.
Frozen validation-state schedule diagnostic `8cce545` shows joint contact XYZ
p95 increases29.33->60.56mm when changing per-tick query to eight-tick plan
reuse/padding; whole-wave PD MSE .02868->.22842. GT padding also changes first8
p95 3.72->17.78mm. This is not closed-loop attribution. Next is the frozen
[one-variable cadence Decision](P-20261011-competent-actor-cadence.md), not a
retraining sweep, nor a negative conclusion about ref3 or the core Cm hypothesis.

## Limitations / future evidence

Same reference and collection seed with env holdout; G is constant reference
endpoint, so no goal-generalization/conditioning claim. Teacher does not reach
G accurately. Oracle GT-R calibration and independent PhysX rows are privileged
controls, not exact hidden-state forks. Tail padding, single-seed bounded Probe
and no PointWorld/critic/RL. Formal validation and varied goals wait for a
competent baseline.

## Resource-only amendment before execution r3

Execution r1 failed before control because a relative checkpoint path crossed
native cwd; f4e1881 fixes it. Execution r2 stopped on a foreign context at its
256-control resource check. Neither is a valid negative or completed result.
GPU inspection shows that process1969748 performs its actual Isaac Sim work on
GPU1 (3682MiB, per-process SM10%); GPU0 is0% aggregate utilization and only166MiB
context, with24076MiB free. Permit this explicitly recorded pre-existing idle
context on GPU0 only: launch utilization<=5%, total prior allocation<=512MiB;
stop on new foreign PID or existing foreign context growing past512MiB. Never
kill, modify or request changes to that process. Own expected allocation18.1GiB
leaves>5GiB headroom. This resource refinement is within Campaign limits, changes
no seed/model/scientific variable/gate or budget, and is frozen before r3.
Retain incomplete run identities; future conflicts preserve partial measurements
without allowing them as training data or completed scientific results.
