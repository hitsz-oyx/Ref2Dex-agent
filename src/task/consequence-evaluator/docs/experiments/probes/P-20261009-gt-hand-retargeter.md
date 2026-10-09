---
schema: ref2dex.probe.v2
probe_id: P-20261009-gt-hand-retargeter
experiment_id: P-20261009-gt-hand-retargeter
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: c28d9eb
claim_id: C3
hypothesis_family: HF-gt-hand-retargeter
probe_index_in_family: 1
seed_pool: probe
seeds: [282, 292]
decision_changed_if_positive: retain geometry-action retargeting and design an H-to-hand policy Probe
decision_changed_if_negative: diagnose retarget/control before any hand policy training
status: UNCLEAR
run_id: gt-hand-retargeter-20261009-r2
---

# GT hand geometry to learned native control

Result: UNCLEAR: the execution gate failed; recorded GT-target servo tracks within4.35mm but held117 versus teacher484, while the frozen learned retargeter tracks poorly (135.89mm, held42/repeat0).
Decision: Keep ACT/G paused and do not train H-to-hand or connect PW/evaluator; diagnose contact execution and the learned inverse separately before repeating the gate.

## Motivation and Decision Note

This Decision Probe serves the self-trained manipulation baseline under C3;
the global Cm claim is unchanged. Does future hand geometry plus explicit
current robot state suffice to reproduce sustained grasp? If yes, retain the
intermediate action representation; if no, diagnose retarget/control before
investing in H-to-hand prediction, PointWorld or evaluator integration.

Existing seven full native teacher launches have q/dq, 11-point geometry and
captured controls. Two earliest packets lack environment/actor-batch identity;
exclude those before fitting, retaining five fully identified launches. Their
full-task success is false (terminal settle absent), so this is a sustained-hold
upper-bound Probe, not a complete manipulation success demonstration.

One idle GPU2: <=3000 fit steps/300 seconds, one full542-step four-role native
launch <=150 seconds plus an implementation correction only if needed; total
<=15 GPU minutes, outputs <1GB. No external authorization boundary crossed.
Stop on provenance, control mapping or timing mismatch; preserve failed runs.

## Frozen protocol

Teacher role0 only, full-launch split: train temporal
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
- [Execution audit](../../../tools/audit/audit_hand_retargeter_execution.py)

Fit at `ac66afb`, `gt-hand-retarget-fit-20261009-r2`: 390/130/130 train/val/test
windows, 2400 steps in27.97s on GPU2, peak allocated113MB. Validation chose
step1600, normalized L1 .24078; test .23493, relative-position target RMSE
27.17mm, local-rotvec RMSE .06553rad, active-finger RMSE .05567rad. Zero-hand
normalized test L1 .51031. These are target-fit diagnostics, not executed
hand tracking. All five native label roundtrips max error <=1.79e-7.

Initial pre-training launch `train.log` hit Python3.8 Path API incompatibility
before data/model work; fixed in58c838c. `train-r1.log` then rejected the two
incomplete identities before optimization. Preserve both; revised split was
frozen before the only actual fit, `train-r2.log`.

Initial native r1 stopped before any action on its DOF-name guard: descriptive
ring/pinky names were reversed. Actual Gym asset audit showed indices10/11
pinky and12/13 ring. Numeric indices, scales, coupling, training labels and
weights were correct. `c28d9eb` corrects names. Original best.pt SHA
`132e443bdd7a85c6acb8a300b8877c32803be29c76977a4bdf17c522584a7562` remains intact;
execution.pt SHA `96f5682d496df0bc502f7585c41a7fa689af5e86c56564a236133915fd11a33f`
contains only the descriptive correction, verified bitwise-equal state_dict
and identical statistics in `metadata_correction.json`. r1 is an engineering
preflight failure, not negative behavior evidence.

Commands (from repository root; complete native argv is in r2/run_manifest.json):

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/consequence-evaluator/tools/run/train_hand_retargeter.py --output outputs/consequence-evaluator/gt-hand-retarget-fit-20261009-r2 --gpu 2 --steps 2400 --seconds 300
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/consequence-evaluator/tools/audit/audit_hand_retargeter_execution.py --packet outputs/consequence-evaluator/gt-hand-retargeter-20261009-r2/retarget.pkl --output outputs/consequence-evaluator/gt-hand-retargeter-20261009-r2/audit.json --gpu 2
```

## Limitations / future evidence

Five same-seed launches are correlated engineering data; no task, object or
seed generalization. No complete task-success demonstrations. Fixed GT source
and live teacher can drift under unexposed PhysX state. Learned repeat arms
are solver-drift diagnostics, not independent seeds. Formal multi-seed tests,
full controlled placement and learned H-to-hand policy are deferred.

## Real execution result and attribution

`c28d9eb`, `gt-hand-retargeter-20261009-r2`, seed282, native GPU PhysX/GPU
pipeline, four envs and64 copies/role. Full542 controls,23 queries at0,24,...,528;
worker81.48s (process87.20s), GPU2 PhysX occupancy18.2GB/utilization17-40%.
All full-task successes remain false. GT source held484, live teacher held484;
required held gate is >=435.6frames (integer436), and all-frame coordinate
tracking RMSE<40mm.

| Role | Max held frames | Hand coordinate RMSE mm | Max lift m | Gate |
|---|---:|---:|---:|---|
| Live reactive teacher | 484 | 41.80 | .820 | behavior reference |
| Direct recorded GT PD-target servo | 117 | 4.35 | .453 | fail hold |
| Learned retargeter | 42 | 135.89 | .443 | fail both |
| Same learned target repeat | 0 | 136.05 | .038 | fail both |

Tracking is against one held-out fixed source, not each role's live teacher;
the live teacher can adapt away from that source and retain grasp. Learned
repeat shares the exact absolute target stream but retains its own q-feedback;
held42 versus0 is simulator/contact sensitivity, not independent seed evidence.

The independent audit at9d41d15 passed: actual=requested controls bitwise,
GT geometry is t+1..t+24 with only unused final padding, q/dq/current geometry
match the learned role at each query, repeated target streams identical.
Frozen GPU checkpoint replay relative error6.11e-7 across native Torch2.0.1
and audit Torch2.4.1, relative-to-absolute roundtrip and mechanical conversion
agree. Direct GT servo has zero clipping and commanded-target mismatch
<=5.96e-8. Thus its loss is not explained by a wrong action index, wrist unit,
missing delta feedback or clipping in this run. It had a stable grasp through
tick176, then one unheld-loss event and no recovery, despite low hand RMSE.
This flags a contact/execution limit of this fixed-source replay; it does not
prove geometry actions or future geometry planning are insufficient.

The learned inverse is also inadequate in this Probe: first120-frame hand
RMSE70.25mm with zero clipping; mismatch predates the first clipped target
at tick265. There are349 clipped coordinates per learned role later, with
max full-PD deviation .4891rad; clipping does not explain the earlier grasp
failure. Do not blame clipping alone, or promote offline27.17mm target fit
to successful hand tracking. The relative-target learned model is
UNPROMISING at these weights/data, while the broader retarget/control route
remains UNCLEAR because even recorded-command replay misses the hold gate.

No extra fit, simulation sweep or H-to-V/PW/evaluator training follows this
failed upper-bound gate. Next minimal discriminating work is contact-window
execution diagnosis of the direct GT servo, alongside source-versus-live-state
inverse replay using existing packets; do not change the global claim or Y.

Artifacts: fit `outputs/consequence-evaluator/gt-hand-retarget-fit-20261009-r2/`,
behavior/audits `outputs/consequence-evaluator/gt-hand-retargeter-20261009-r2/`:
`retarget.pkl/json`, `audit.json`, `audit-r2.json`, `diagnostics.json`,
`tracking_vs_grasp.png`, `native.log`, `run_manifest.json`. Initial engineering
logs/asset audit remain inr1. New outputs total<25MB; GPU2 released.13 focused
retarget/action-chunk tests passed. Repository/index verification recorded
in the final local commit.
