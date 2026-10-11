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
status: PLANNED
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

Pending.

## Limitations / future evidence

Same reference and collection seed with env holdout; G is constant reference
endpoint, so no goal-generalization/conditioning claim. Teacher does not reach
G accurately. Oracle GT-R calibration and independent PhysX rows are privileged
controls, not exact hidden-state forks. Tail padding, single-seed bounded Probe
and no PointWorld/critic/RL. Formal validation and varied goals wait for a
competent baseline.
