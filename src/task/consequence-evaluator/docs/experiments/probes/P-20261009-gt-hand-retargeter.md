---
schema: ref2dex.probe.v2
probe_id: P-20261009-gt-hand-retargeter
experiment_id: P-20261009-gt-hand-retargeter
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-gt-hand-retargeter
probe_index_in_family: 1
seed_pool: probe
seeds: [282, 292]
decision_changed_if_positive: retain geometry-action retargeting and design an H-to-hand policy Probe
decision_changed_if_negative: diagnose retarget/control before any hand policy training
status: UNCLEAR
run_id: gt-hand-retargeter-20261009-r1
---

# GT hand geometry to learned native control

Result: UNCLEAR: execution pending.
Decision: Pause ACT inference and action-to-hand bridge per user ref7; first test GT-hand retarget/control only.

## Motivation and Decision Note

This Decision Probe serves the self-trained manipulation baseline under C3;
the global Cm claim is unchanged. Does future hand geometry plus explicit
current robot state suffice to reproduce sustained grasp? If yes, retain the
intermediate action representation; if no, diagnose retarget/control before
investing in H-to-hand prediction, PointWorld or evaluator integration.

Existing seven full native teacher launches have q/dq, 11-point geometry and
captured controls. Reuse these rather than collecting new fork panels. Their
full-task success is false (terminal settle absent), so this is a sustained-hold
upper-bound Probe, not a complete manipulation success demonstration.

One idle GPU2: <=3000 fit steps/300 seconds, one full542-step four-role native
launch <=150 seconds plus an implementation correction only if needed; total
<=15 GPU minutes, outputs <1GB. No external authorization boundary crossed.
Stop on provenance, control mapping or timing mismatch; preserve failed runs.

## Frozen protocol

Teacher role0 only, full-launch split: train native-chunk r2/r3, temporal
open_loop24/receding8/overlap8; val temporal1; test receding1. All seed282,
same actor/controller/backend/environment. Fit seed292, stride4 complete24
windows, no reset crossing or padding. Train statistics only, per-horizon
geometry/action normalization; checkpoint selected only on val L1.

Model inputs: future world-frame 11-point hand displacement at t+1..t+24
relative to measured current hand, and current q18+dq18. Outputs: recorded
commanded PD targets at t..t+23 relative to query-time q, position in meters,
local SO(3) rotation vector, six independent finger angles. Wrist URDF order
is intrinsic XYZ; native DOF order is explicitly checked. Two-layer, four-head
Transformer decoder, width128, learned horizon queries, normalized L1.

Native wrist actions are increments, so each frozen absolute PD target is
converted using live mechanical q; this is fixed-target servo feedback, not
24 raw native increments without feedback. R is queried every24 ticks. Last
query528 has14 usable futures: unused10 terminal geometries repeat for model
input only and are never executed or used as training labels.

Roles: reactive teacher; direct recorded GT commanded-target servo; learned
retargeter; repeat of the same learned absolute target stream. Role1 diagnoses
the native conversion upper bound; source future q/actions enter that role
and offline labels only, never learned-model inputs. GT future geometry is
privileged, so this is not a deployable planner. No strict same-state claim.

Predeclared sustained-hold gate: live teacher held>=45; direct GT target servo
held>=90% of teacher and hand coordinate RMSE<40mm; then both learned roles
held>=90% of teacher and RMSE<40mm over all542 future frames. Report source
teacher held and all full-task outcomes separately. The40mm threshold is a
Probe screen for clear improvement over the prior ~63mm bridge, not a matched
scientific comparison. Offline action/target error cannot pass the gate.

## Implementation and evidence

- [Relative target model](../../../src/consequence_evaluator/retargeter.py)
- [Teacher-only training](../../../tools/run/train_hand_retargeter.py)
- [Native runner](../../../tools/run/run_gate1_gt_progress.py)
- [Contracts](../../../tests/test_retargeter.py)

## Limitations / future evidence

Seven same-seed launches are correlated engineering data; no task, object or
seed generalization. No complete task-success demonstrations. Fixed GT source
and live teacher can drift under unexposed PhysX state. Learned repeat arms
are solver-drift diagnostics, not independent seeds. Formal multi-seed tests,
full controlled placement and learned H-to-hand policy are deferred.
