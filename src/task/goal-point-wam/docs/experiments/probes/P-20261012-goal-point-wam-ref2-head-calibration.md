---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-head-calibration
experiment_id: P-20261012-goal-point-wam-ref2-head-calibration
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 7f053ac9
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 17
seed_pool: probe
seeds: [290, 291]
decision_changed_if_positive: retain explicit robot-domain head adapter and review one bounded simulation comparison
decision_changed_if_negative: inspect finite-teacher coverage and frozen R dependence before expanding robot execution
status: UNCLEAR
run_id: wam-ref2-head-calibration-20261012-r1
---

# Robot-domain trajectory head calibration with frozen WAM scene and R

Decision: distinguish whether a small explicit robot-domain head adaptation can
substantially reduce the WAM->R command mismatch, or the retained teacher data
and R interface are still inadequate. This serves the autonomous trajectory
robot baseline subgoal; it is not RL, video training or a Cm-on/off comparison.
The preceding fixed [shadow](P-20261012-goal-point-wam-ref2-shadow.md) passed
numeric frame/index/PD wiring but failed PD-distance gates. Its64-window
GT-R/source true PD XYZ p95 is10.59mm; don't infer impossibility merely from a
10mm generic screen. Its human checkpoint remains unchanged.

## Fixed recipe and resource bounds

Seed290 for Torch; draw291 for training batches. One idle GPU2, <=15min
including engineering backward smoke, cache and <=300 updates. All new ref2
artifacts including shadow <=100MB; head-only final checkpoint<10MB, no full
WAM checkpoint/optimizer/cached-token disk output. Scene/PTv3, goal encoder,
executed-action/world branches and R are frozen. Train only existing trajectory
decoder and observed-hand query adapter, initialized from original final10000.
Authenticate original source e8b81994 and all shadow input identities.

Use prior qualified32-teacher data and its fixed train env0:16, validation16:24.
No test env28:32 is read for feature/label construction. Frozen64-window
env24:28 query panel from the earlier shadow is reported only at initial/final,
never for checkpoint/weight/threshold selection. Every complete t+24 window
is eligible (519*16 train and519*8 validation). Inputs remain measured H4 and
planned reference object goal at t+24; measured future hand and true PD are
labels only. Startup repeated current-state padding is retained and reported.

Frozen scene/goal tokens cached only in RAM/GPU; trainable observed-hand
adapter reads original start-hand XYZ/history with original train-only WAM
normalization. First verify cached input produces the same baseline prediction
as direct inference to <=1e-5m, finite backward enters both trainable modules,
and frozen weights have no gradients. AdamW lr3e-4, weight_decay.01, batch128,
clip_grad1, BF16 actor, R forward in float32 with gradient to predicted tau.

Loss is1*mean Huber(predicted-current-relative24-step right-hand points /
.05m, true points /.05m) +1*mean Huber(R encoded independent12-D PD target
/ frozen R target_scale, actual encoded true PD / frozen R target_scale).
No future measured object/q/contact input, no teacher intervention at evaluation.
The objective uses real24-step robot hand labels and real commanded PD labels,
not R's own GT prediction as a fitted label. Fixed final300 only; no best-model
selection or model sweep. Stop on nonfinite, foreign GPU PID,15min wall time,
artifact bound or frozen-weight/source drift. Evaluate initial and final using
fixed validation64 windows (same ticks/env16:20) and the previously declared
held64 env24:28. Train sampled windows may not come from these held envs.

Predeclared signal: >=50% reduction in held64 first8 EPE and WAM-vs-GT-R PD
XYZ p95, with no increase in clipping. Also retain original conservative
simulation admission screen (PD XYZ<=10mm, wrist/finger max-coordinate
p95<=.05rad, clipping<1%, frame wiring passed). Passing adaptation signal
without full admission means PROMISING for robot adaptation, with simulation
still deferred. Report true-PD deviation against both original GT-R baseline
and adapted WAM-R, rather than relabeling the retained R's finite precision.

## Results

Engineering one-step smoke completed12.71s, GPU2 peak656.34MiB; cached/direct
prediction max error0, trajectory/hand_query gradient sums5569.70/324.36,
all five frozen module hashes unchanged and no frozen parameter gradients.
Retained record `outputs/trajectory-policy/wam-ref2-head-smoke-20261012-r1/`.
This verifies wiring only; no checkpoint or scientific adaptation conclusion.

After smoke, the formal entry additionally validates positive teacher
readiness, training_allowed, audit.pass and dt1/30 before loading teacher
arrays; monitoring guards combined new ref2 outputs and its private scratch
against100MB. It records actual startup draw counts and initial/final history
strata without changing uniform weights. New128-cache/64-eval batch context
causes roughly.06mm mean full24 EPE difference from the original8-batch
shadow; initial/final training comparisons reuse exactly the same frozen cache.

Formal calibration at7f053ac9 completed300 updates in61.42s (cache38.06s,
head optimization about20.4s), GPU2 peak2459.65MiB. All five frozen hashes
exactly unchanged; cache/direct max0 and both trainable modules have nonzero
gradients. Native GPU sampling showed0% instantaneous utilization while
memory and CUDA compute were active; this bounded head-only test does not
justify spending extra compute to force utilization.

Held64 initial/final nearest8 EPE86.53->28.03mm, full24 205.41->38.21mm,
h1 12.10->9.11mm. WAM-vs-GT-R XYZ p95288.49->199.83mm, rotation
1.095->.652rad, fingers.716->.399rad; clipping remains0. Actual PD deviation
final XYZ198.69mm versus GT-R baseline10.597mm; report R's finite precision
separately. First8 reduction criterion passes but >=50% PD reduction does not,
so predeclared joint signal remains UNCLEAR. All three distance screens fail;
the passing frame/semantic/PD conversion gates remain intact.

The12 startup repeat-H4 report rows have first8 EPE180.95->59.45mm,
h1 31.72->28.62mm and PD XYZ p95311.99->225.53mm. The52 real-H4
rows have first8 64.74->20.77mm, h1 7.58->4.60mm and PD XYZ
155.07->34.88mm. Later-than-tick8 first8 is18.68mm/h1 3.84mm,
PD XYZ p9535.77mm. Startup draws240/38400=.625% follow uniform sampling;
startup rows12/64=18.75% of the intentionally startup-rich report panel.
No held-driven reweighting or checkpoint selection occurred.

Validation64 env16:20 has first8 87.13->27.27mm, full24
206.90->36.20mm, PD XYZ p95288.54->199.18mm, clipping0.
Raw initial/final prediction/teacher/PD/action arrays, all frozen hashes and
draw counts are retained at
`outputs/trajectory-policy/wam-ref2-head-calibration-20261012-r1/`.
Head-only checkpoint SHA1d1d9678bf7788d2929677c6e93f653a21126656fb5d56641c5c5017a31a9724;
no base WAM, R, optimizer or disk token cache was written.

User ref2 specifically requests minimal simulated closure after interface
gates to determine actual control value. Root has recorded a separate Decision
allowing a bounded3-row teacher/GT-R/WAM-R simulation despite the failed
conservative offline screen. This preserves this card's UNCLEAR result and
screen; no oracle prefix, frame0 shortcut or physical robot execution is allowed.

## Limitations / future evidence

All episodes share one nominal motion and closely repeated teacher states;
held envs are not arbitrary new physical initial states. Task-specific head
learning does not prove robot-domain generalization, native human backbone
retention, closed-loop feasibility, real robot transfer or Cm utility. If
promising, separately check human-domain retention and original native runner
deployment contract before any small simulation. Preserve original checkpoints
and all raw run records regardless of result.
