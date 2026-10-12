---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref26-gated-hand
experiment_id: P-20261012-goal-point-wam-ref26-gated-hand
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 161d4b06
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 36
seed_pool: probe
seeds: [295, 296]
decision_changed_if_positive: retain warm-start preserving hand adapter for subsequent staged study; C remains paused in this revision
decision_changed_if_negative: preserve baseline and qualified Chunk-R; stop this short adaptation without rejecting the hand representation
status: PROMISING
run_id: wam-hand-gated-fit-20261012-r1
---

# ref2_6 B2: zero-init gated hand residual

Result: Initial outputs exactly preserved; matched200 updates pass21/21 checks. Nominal/startup first-PD p95 improve11.30%/18.04%; direct hand-feature utility remains UNCLEAR after ablation.
Decision: Retain B2 as a safe adapter candidate and both packets; preserve qualified A2 Chunk-R. C stays paused; do not attribute most paired gains to the live hand residual.

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

## Results / independent attribution

Implementation/training commit161d4b06, audit subsequently checks persisted
patches independently against the original warm packet. Smoke31.34s; paired
scientific run100.69s. Initial predictions, R first controls and shifted-goal
outputs are exactly identical across all five panels. Initial alpha0;
update1 gate gradient.095703125, projection gradient0 as designed; update2
projection gradient.0260003. Final alpha+.0001711579. R/action/world/scene/
goal/raw adapter and every normalization buffer have zero recorded change;
independent saved scene/goal/hand_query comparisons are bitwise equal to warm.

| Panel | Old h1 / first8 mm | B2 h1 / first8 mm | Old first-PD XYZ p95 mm | B2 first-PD XYZ p95 mm |
| --- | ---: | ---: | ---: | ---: |
| Nominal report | 6.163 / 15.421 | 6.168 / 15.420 | 19.600 | 17.386 |
| Startup report | 6.627 / 11.462 | 6.625 / 10.829 | 11.088 | 9.087 |
| t8 recovery TRAIN | 5.522 / 13.695 | 5.814 / 13.719 | 16.479 | 17.232 |
| t16 recovery TRAIN | 5.903 / 15.405 | 5.902 / 15.347 | 21.999 | 21.231 |
| t32 recovery TRAIN | 11.401 / 23.034 | 12.465 / 22.260 | 143.698 | 145.698 |

21/21 non-regression checks pass; all clipping0. Goal response76.751->72.889mm
passes preservation threshold. Relative to matched decoder-only training,
nominal/startup first-PD improve11.30%/18.04%; startup first8 improves5.53%.
Recovery t32 h1 worsens9.33%, still inside the predeclared10%+.05mm tolerance;
passing this tolerant screen does not mean every metric improved. Nominal
all24 increases21.689->22.189mm. Do not substitute Probe35's all-policy-trained
control (6.51mm h1) for the contemporaneous frozen-encoder control here.

Post-fit alpha0 ablation leaves nominal PD17.493mm, startup PD9.299mm and
startup first8 10.838mm: most paired improvement persists without live hand
injection. The direct gate effect is only.61%/2.27% on these control metrics;
h1/first8 changes are tiny. Nonzero alpha and gradients prove use of the
interface, not useful hand representation. **PROMISING applies to warm-start
preservation and this offline adaptation screen; PTv3 hand-feature utility
remains UNCLEAR.** The two training paths may alter decoder optimization;
this one run cannot establish which mechanism caused the paired improvement.
No autonomous grasp/control qualification or formal representation claim.

Independent audit recomputes180 labels/metrics including the alpha0 ablation,
all deterministic draws,21 thresholds and the15 useful-signal comparisons;
maximum tau label error1.192e-7m, action/metric errors0. Saved frozen patches
and alpha also verified. Five hand-gather/gate tests and six Chunk objective/
retargeter tests pass; root changed-file verification passes.

- [Matched result](../../../../../../outputs/trajectory-policy/wam-hand-gated-fit-20261012-r1/result.json)
- [Independent audit](../../../../../../outputs/trajectory-policy/wam-hand-gated-fit-20261012-r1/root_hand_readout_audit.json)
- [Run protocol](../../../../../../outputs/trajectory-policy/wam-hand-gated-fit-20261012-r1/manifest.json)

Fit423.24MB + smoke8.72MB + scratch1.89MB =433.84MB<500MB. Global outputs+tmp
293.70GB<400GB (du counted separately from the owned artifact directories).
GPU2 sampling utilization0-3%, VRAM peak sampled1581MiB; short-run throughput
~.18-.20s/update. Low utilization was reported; no recipe changed mid-run,
no GPU throughput claim. GPU2 is released. Qualified Chunk-R remains intact.

## Limitations / future evidence

Single deterministic short Probe; nominal windows share motion/reset and
recovery is training support. No formal hand-representation verdict, physical
actor utility, multi-seed Validation, or learned world feedback. Teacher-forced
world implementation is preserved but not trained/evaluated by this probe.
