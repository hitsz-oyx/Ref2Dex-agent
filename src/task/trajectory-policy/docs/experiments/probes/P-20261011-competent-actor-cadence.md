---
schema: ref2dex.probe.v2
probe_id: P-20261011-competent-actor-cadence
experiment_id: P-20261011-competent-actor-cadence
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 8cce545
claim_id: C3
hypothesis_family: HF-trajectory-policy-competent-actor-cadence
probe_index_in_family: 1
seed_pool: probe
seeds: [184]
decision_changed_if_positive: retain per-control measured feedback trajectory BC and reassess baseline readiness
decision_changed_if_negative: investigate closed-loop actor distribution before Cm or new hyperparameter training
status: PLANNED
run_id: competent-joint-bc-cadence-20261011-r1
---

# Does current-state per-control trajectory replanning preserve competence?

## Motivation / Decision

The [matched joint BC](P-20261011-competent-joint-bc.md) full execution r3 is
UNPROMISING: teacher and oracle R4/4 each, autonomous tau_only/joint0/8 each.
Actual H/G/trajectory/world plan/R/native PD replay passes. First actions are
near calibration, then the approach deviates before first grasp. No PointWorld.
Frozen validation-state schedule replay `8cce545` (no refit/test selection) shows
joint contact PD XYZ p95:period1 29.33mm versusperiod8 60.56mm; whole-wave MSE
.02868 versus.22842. GT period8 itself raises first8 p95 from3.72 to17.78mm.
Thus the offline training interface (query at every tick) differs from deployed
shifted/padded24-step conditioning. This is a plausible contributor, not proven
closed-loop cause. A one-variable frozen physical replay is cheaper and more
informative than more epochs, architecture or RL before a competent baseline.

## Fixed protocol

Use the exact retained matched actor checkpoints:
- tau_only SHA4db89b573f24e2ea9c756941759a627716af5af8cf37b3841ce6ef6ae0afb282
- joint SHA9527250b9959dc2f0aa5d27ae55a812f82a6a7833da8700efb8f035bf6380d92
- R SHA414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f

Same seed184 and24env layout:teacher4, GT-tau/R4, tau_only8, joint8. Same native
reset/controls542/goals/metric/thresholds and source donor24:28. Change only
actor query period8->1: fresh own measured H/G generates24 future hand each
control, R uses its own live state. No teacher plans/actions enter autonomous
arms, no phase/clock/future q/object, no PD feedforward or pretrained teacher
base. No new fit, checkpoint choice, restart from favorable states or selection
on physical outcomes. Record full per-tick query/plan/PD and independent replay.
Calibration and joint gates exactly as prior card:teacher+GT each>=3/4 hold45
and controlled-placement, joint>=6/8 both, median held>=80% GT, clipping<1%.
All pass->PROMISING; valid calibration plus joint fail->UNPROMISING;
invalid calibration->UNCLEAR. Tau_only separately reported. Improvements relative
to prior r3 are local cadence evidence, not a formal matched multi-seed claim.

## Resources / stops

Shared original route20GPUmin/1GiB remaining envelope, not added budget. OneGPU0,
physics<=300s, audit<=120s, outputs<=256MiB. The already recorded foreign idle
context PID1969748 may remain<=512MiB, aggregate prelaunch<=5% and512MiB; stop
on a new PID, context growth>512MiB, nonfinite/reset/input drift or timeout.
Never operate on the other user's process. No Campaign change/branch/push.

## Results

Pending.

## Limitations / future evidence

Same reference/reset distribution with environment holdout; variable G and
accurate object target attainment untested. Querying every tick adds inference
cost and alters feedback response, so it cannot isolate padding alone. No formal
validation or Cm/RL benefit. Successful grasp without correct reference following
is not Mission completion.
