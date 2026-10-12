---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref26-gated-hand
experiment_id: P-20261012-goal-point-wam-ref26-gated-hand
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 36
seed_pool: probe
seeds: [295, 296]
decision_changed_if_positive: retain warm-start preserving hand adapter for subsequent staged study; C remains paused in this revision
decision_changed_if_negative: preserve baseline and qualified Chunk-R; stop this short adaptation without rejecting the hand representation
status: UNCLEAR
run_id: wam-hand-gated-fit-20261012-r1
---

# ref2_6 B2: zero-init gated hand residual

Result: Protocol fixed; engineering smoke and matched short fit pending.
Decision: Change only hand injection; preserve qualified A2 Chunk-R and pause C.

## Decision Note / motivation

This Decision Probe serves C3 robot manipulation. Probe35 hard-add disturbed
warm nominal h1 from6.94 to54.51mm before fitting, so its negative short-fit
result confounds feature utility with interface adaptation. User ref2_6
requests raw + alpha P(LN(full-packed semantic current-hand tokens)), alpha0.
Retain the qualified Probe34 Chunk-R (GT74held), backbone, goal32 and world
interface. The cheapest discriminator is one matched offline200-update fit,
preceded by actual model-output equality checks. No new physics or C feedback.

Separate subclass preserves historical module keys. LN128 and two-layer
128-to128 GELU projection have ordinary nonzero initialization; only scalar
alpha is zero, avoiding dead projection gradients. Mask missing hand slots
after projection to eliminate bias-created observations. Raw adapter remains.
All current2x11 tokens come from full packed PTv3 features before sampling.
Initial tau, first R controls and +/-20mm shifted-goal outputs must be exactly
equal across arms on all five report panels; otherwise abort before B2 fitting.

Both arms load Probe32 warm policy and fixed Probe34 R, model295/draw296.
Freeze PTv3 scene, goal, raw hand_query, action/world, buffers and R. Train only
trajectory in the control, trajectory + LN/projection/alpha in B2. AdamW1e-4,
weight decay.01, clip1, eval-mode deterministic encoder. Identical200 updates,
batch64=16startup+16later nominal TRAIN+32 actual recovery futures (11/11/10).
Same true tau MSE/.05 + control-aware frozen-R loss as Probe35. First gate
gradient must be nonzero; projection gradients must wake on update2. CPU
unit tests cover tiny wiring only; full fit/inference uses physicalGPU2.

Same five panels and21 non-regression checks as Probe35: h1/first8 <=1.1old
+.05mm; first native PD XYZ p95 <=1.1old+.1mm; clip<=old+.01; cached-scene
goal response >=max(.001mm,.5old). Report all24 as secondary. Passing these
alone is safe adaptation, not feature benefit: predeclare useful signal as
all21 passing plus >=5% improvement in at least one panel's h1/first8/firstPD.
Non-regression without useful signal is UNCLEAR; failure is UNPROMISING for
this short recipe. Post-fit alpha0 ablation with the identical trained decoder
recomputes all panels, supplying inexpensive attribution without more fitting.
Recovery panels remain TRAIN; isolated metric gains are exploratory only.

Budget supplement: sequentialGPU2, paired1200s/20GPUmin, separate cumulative
B2 outputs + smoke + tmp/wam-hand-gated <=500MB; global outputs+tmp <=400GB.
No intermediate checkpoints; retain full old/B2 policy patches and predictions.
Stop on frozen drift, nonfinite, missing semantics/gradient, unequal initial
outputs or training draws, foreign GPU2 process, deadline or artifact limit.
This does not reopen the backbone or core claim. C remains paused regardless
of this run's classification; any next-stage protocol is separate work.

## Implementation / reproduction

- `src/task/goal-point-wam/src/goal_point_wam/hand_gated.py`
- `src/task/goal-point-wam/tests/test_hand_gated.py`
- `src/task/trajectory-policy/tools/run/probe_wam_hand_readout.py --gated`
- `src/task/trajectory-policy/tools/audit/audit_wam_hand_readout.py`

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/trajectory-policy/tools/run/probe_wam_hand_readout.py --gated --engineering-smoke --output outputs/trajectory-policy/wam-hand-gated-smoke-20261012-r1
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/trajectory-policy/tools/run/probe_wam_hand_readout.py --gated --output outputs/trajectory-policy/wam-hand-gated-fit-20261012-r1
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/trajectory-policy/tools/audit/audit_wam_hand_readout.py --fit outputs/trajectory-policy/wam-hand-gated-fit-20261012-r1
```

## Limitations / future evidence

Single deterministic short Probe; nominal windows share motion/reset and
recovery is training support. No formal hand-representation verdict, physical
actor utility, multi-seed Validation, or learned world feedback. Teacher-forced
world implementation is preserved but not trained/evaluated by this probe.
