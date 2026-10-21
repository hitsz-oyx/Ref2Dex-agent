---
schema: ref2dex.probe.v2
probe_id: P-20261022-wam-v6-flowmatching
experiment_id: P-20261022-wam-v6-flowmatching
date: 2026-10-22
task: ptv3-shared-wam-v6
branch: main
git_commit: 2d8bf18f347b9f871e1c4a6b4bf589740688a6d2
claim_id: C1
hypothesis_family: HF-v6-joint-trajectory-flow
probe_index_in_family: 1
seed_pool: probe
seeds: [170,171,172]
decision_changed_if_positive: retain residual conditional flow if fixed held-out short-prefix errors improve stably
decision_changed_if_negative: retain checkpoints and inspect generalization before scaling the flow branch
status: PREPARING
run_id: flowmatching-overnight-20261022-r1
---

# Joint hand/object conditional residual flow with GT-object contact

Result: implementation and unit contracts prepared; real-input and three-rank trial pending.
Decision: owner explicitly requests longer training after the runnable trial; efficacy remains open.

## Question and minimal decision probe

Does a joint hand/object conditional vector field learn useful trajectory residuals
around healthy WAM-v6 predictions, retaining the latest GT-object contact loss?
This Decision Probe first tests real-input learning, noise-only ODE sampling and
strict distributed recovery; failure blocks the long run. Success permits the
requested longer exploration, not a claim that FM solves rolling drift.
Compare fixed DEV prefix1–8 hand/object errors, horizon24 object error, contact
and feature-stability diagnostics with initialization.

## Method and retained objective

Healthy no-geometry v6 source:
`outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/training/step004371.pt`.
Copy all old weights and four normalizer buffers exactly; new flow parameters,
Adam/RNG/counters start fresh. Keep goal/next-event conditioning,24 future ticks,
joint21 per hand,32 object points and original coordinate/C/part supervision.
This expands the architecture, not strict resume of GT-contact3238; later
trial-to-long transitions restore optimizer/RNG/config strictly.

One future-tick token packs both hands and object (222 coordinates). New width256,
two-layer temporal Transformer decoder attends to observed shared context.
Linear training path `x_t=(1-t)*noise+t*(GT_normalized-base.detach())`, velocity
`GT_normalized-base.detach()-noise`, residual noise sigma0.1. Generate actual
coordinates with four differentiable Euler steps from noise, add to pretrained
base. Coordinate/contact losses see this target-free ODE sample, never the
teacher interpolation. Backbone encoded once. Fixed sample-ID/seed172 DEV noise;
alternate seed/eight-step audits. Future GT labels remain inference-inaccessible.

Strict FP32 PTv3/attention/flow, no AMP/TF32. Shared/geometry LR1e-5, new flow
LR1e-4, warmup50, clip1. Keep normalized tau and motion-weighted scene MSE,
learned C BCE/part CE; add masked hand/scene velocity MSE weight1. This hybrid
retains pointwise MSE and cannot establish multimodality by itself. Learned C
head is still observed-feature conditioned, not resampled-trajectory conditioned.
TRAIN total includes velocity terms; DEV total excludes teacher-only terms.

Contact unchanged: GT future ordered2048 object points plus predicted-tau hand
skin, proximity probability vs GT labels; ticks1/4/8/12/16/20/24,chunk64,
weight0.05/ramp100,clearance0.25. GT object is loss-only. This term directly
constrains generated hand, not predicted object; scene/flow losses supervise it.

## Inputs, resources and execution plan

Recipe `outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json`
and sibling stats:207579 TRAIN/3356 DEV windows,1177 caches,~99.8% GRAB draws.
Longer training does not recover the full remote nine-source data. Matched parent
seeds170/171, flow172, not independent replication. Runtime
`/home2/wyy/miniconda3/envs/graspenv/bin/python`,Python3.8/Torch2.4.1+cu121.

Only physical RTX3090 GPU4/5/6. Capacity8 onGPU4 at local96; fallback80/64 only
for OOM or reserved>21.5GiB. DDP8/save+strict2/recovery, continuation to128,
then real ODE/teacher-field audit. Automatically continue same optimizer/RNG to
30000 updates or43200s cumulative new-objective training time, whichever first.
DEV0/20/128/every500, checkpoint every500, two recent plus selected within8GiB
per stage. Actual throughput sets final steps.08:00 is an interim review, not
stop time; likely finish ~12h after main stages begin. Absolute supervisor wall
bound47000s,stall20min,disk20GiB floor. Free~175GiB,projected extra<24GiB.
No deletion/migration required. Owner exception recorded in root CAMPAIGN.

Supervisor and children use separate sessions, stdin closed, nohup launch.
Environment allowlist; no credential reads/writes. Frozen tracked code;
third_party symlink read-only. Exact launch/PID/SID/telemetry to be appended.
STOP file requests saving/stopping only this run; no other jobs/cards touched.

## Sources and limitations

Linear conditional flow formulation: primary [Flow Matching paper](https://arxiv.org/abs/2210.02747).
Old `src/task/PointWorldWAM/train_wam_v1.py` has a linear velocity-target example;
its action/MANO contracts are not transplanted. Prior
`P-20261015-mse-multimodality-audit` did not train FM or establish multimodality
as this model's failure cause. No universal FM superiority or drift-fix claim.

Implementation validation:47 v6 tests PASS in80.70s, including target isolation, fixed sampler RNG, differentiable Euler and old-weight-preserving expansion.
