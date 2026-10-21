---
schema: ref2dex.probe.v2
probe_id: P-20261021-wam-v6-gt-object-contact
experiment_id: P-20261021-wam-v6-gt-object-contact
date: 2026-10-21
task: ptv3-shared-wam-v6
branch: main
git_commit: 39d66051a8485c94a15bacf68530e696b1ac201e
claim_id: C1
hypothesis_family: HF-v6-gt-object-contact
probe_index_in_family: 1
seed_pool: probe
seeds: [170,171]
decision_changed_if_positive: retain GT-object geometric contact supervision if held-out short-prefix hand and object errors improve without shared-feature instability
decision_changed_if_negative: inspect predicted-hand proxy fidelity and task-generalization gaps before scaling contact supervision
status: RUNNING
run_id: gt-object-contact-20261021-r1
---

# GT2048 object and predicted-hand contact probability supervision

Decision Probe requested and clarified by the user: stop the current FP32 run,
then use complete2048-point GT future object geometry with predicted tau to
compute contact probability and align it to GT contact labels. User explicitly
selected this over the old relative-vector loss, whose object terms would cancel
if both sides used GT object geometry. This is a new objective, not strict resume.

Result: old run saved/stopped3337;44 tests, real gradient preflight, capacity96, world3 save8 and strict recovery2 PASS; sustained training RUNNING.
Decision: launch a bounded healthy4371 FP32 arm after real GPU preflight.

## Fixed data and inputs

Same healthy no-geometry source step4371 from
outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/training/step004371.pt.
Copy all model weights and4 normalizer buffers exactly; fresh Adam/RNG/count/
warmup as in the previous geometry arms. The user-stopped FP32 step3337 remains
preserved as evidence; it is not the new initializer. Same expanded local recipe,
207579 TRAIN/3356 DEV windows over1177 existing caches and seeds170/171.

A task-owned opt-in reader transforms the same ordered2048 CAD contact queries
by recorded GT future object poses in the current object frame. Labels cover
all24 future frames, float32, without32-flow prediction or affine interpolation.
Only loss code reads geometry_object_future_gt; the model's observed geometry,
goal/event inputs and all parameter keys remain unchanged. Reader defaults off
for old arms and preserves the flag through spawned DataLoader reconstruction.
No raw/cache/stat mutation or future-object network input. Old tau/scene/C/part
losses and goal/event conditioning retained. GT object is unavailable at inference
and is not needed there; this is supervised training only.

## Geometric contact objective

Observed778-point hand skin remains attached to joint21 bones and transported
by predicted tau. At the inherited7 time indices1/4/8/12/16/20/24, evaluate
**all2048 ordered object queries**, no GT top-k or subsampling. Detached nearest
skin indices followed by differentiable selected distances avoid storing a huge
cdist autograd graph. Chunk64 bounds distance workspace; query coverage is not
reduced by chunking. Per-query scalar proximity, not a new learned contact head.

For distance d, radius r=3mm, falloff s=10mm and smoothing b=1mm:

x = softplus((d-r)/b) * b/s; C_geometry = exp(-x).

BCE with the existing soft GT contact label y is computed as
`y*x - (1-y)*log(-expm1(-x))`. The1mm smoothing approximates the native3mm
plateau; the training tail is not truncated at33mm. Evaluating BCE from log
probability preserves a pulling gradient for far-away positives even when the
FP32 probability underflows to0. This changes the probability transfer slightly;
GT labels are unchanged. Unknown future/joint/contact labels are masked.

Keep the prior GT-referenced local noncontact clearance term, now evaluated
against the GT object, weight0.25. It supplies a separating direction at exact
zero-distance overlap, where unsigned distance alone has zero derivative.
Both geometry terms directly supervise the predicted hand, with no direct
scene-readout/decoded-object gradient. Shared-feature updates can indirectly
change scene predictions, and original scene supervision remains active.

Four real contact-positive TRAIN samples calibrate the initial combined hand/
object readout-gradient ratio as before: weight=min(0.05,max(0.0001,
0.5*base_readout_norm/geometry_hand_readout_norm)); then freeze and ramp100.
Require >=80% positive coverage, finite nonzero hand/readout gradients,
exactly absent direct object geometry gradients and absent old C-only readout
gradients. No DEV-based weight tuning. Report GT-tau proxy contact MAE as a
reference for the approximation's discrepancy from actual cached contact labels.

## Resources, bounds and comparison

Same strict FP32 implementation as the previous arm: all parameters/attention
float32, AMP and matmul/cuDNN/spconv TF32 disabled. Python graspenv3.8.20,
Torch2.4.1+cu121/CUDA12.1;8 spawn workers/rank, OMP/MKL2/OPENBLAS1,
NCCL_P2P_DISABLE=1. Previously authorized local GPUs4/5/6, three RTX3090;
unrelated0–3/7 work untouched. Initial free disk>180GiB; combined bounded
checkpoint stages expected<20GiB, stop below20GiB free (>user15GiB hard floor).
No deletion or migration needed.

Capacity8 real disposable updates at local96; fall back80/64 only on OOM or
reserved peak>21.5GiB. Main weights reinitialize from4371. Three-rank readiness8
save, strict2 recovery save, then sustained continuation; audit568 finite states,
step counts, RNG3/world3/effective batch and exact normalizers at each boundary.
10000 updates or14400 cumulative new-objective seconds, whichever first;
LR1e-5/warmup50/clipping1 unchanged. Per-stage checkpoint budget8GiB.
Root STOP requests saved termination; bounded stage/no-progress monitoring.

DEV before updates,20 then every500,32 fixed examples/source/microbatch4.
Preserve previous predicted-hand/predicted-object geometric diagnostics via a
separate evaluation-only branch, so old geometry/contact_relative stays comparable.
New metrics have geometry_gt/ names: contact MAE, near-contact MAE, GT-tau proxy
MAE, positive coverage and noncontact violation. Compare hand/object prefix1–8
mean EPE and h08, rather than only object h24. Available-source macro results
and the GRAB-dominant sample distribution must be reported separately.

This intervention changes object source, dense query coverage, loss family and
TRAIN-calibrated weight together. It tests the user-requested package; a gain
cannot isolate which ingredient caused it. Same ancestor/seeds, not independent
replication. The hand surface remains an approximate joint-based proxy; no exact
mesh collision, grasp physics, rollout success or GT-zero contact claim.

Outputs: outputs/ptv3-shared-wam-v6/gt-object-contact-20261021-r1/.
Frozen code/logs: tmp/gt-object-contact-20261021-r1/.
Entrypoint: tools/run/launch_strict_fp32_training.py --gt-object-contact.

## Engineering evidence

44 tests PASS in11.05s. Tests include direct recorded-pose2048 rotation/translation
and ordering, invalid-pose rejection, actual spawned reader reconstruction,
future-label changes leaving all model outputs exactly unchanged, positive
hand gradients with no direct object/GT-label gradient, unchanged geometry loss
when predicted scene is shifted10m, noncancellation when GT object moves,
complete2048 query coverage, chunk32/64 value/gradient agreement, masked-NaN
safety and finite far-positive gradients to10m with zero probability underflow.
Four real TRAIN reader samples preserve all old inputs/targets exactly;
receipt under gt-object-contact-preparation-20261021-r1/reader-preflight.json.

## Actual three-card launch

Supervisor1572421, frozen39d66051. Healthy source4371 checkpoint audit PASS,
568 finite states/world3/global288/RNG3; all weights/scales copied with a fresh
optimizer only at initialization. Four real TRAIN contact samples exercise
4×7×2048=57344 valid geometric contact queries, coverage100%. Base readout norm
1.228660, geometry hand readout norm10.613400,
new direct decoded-hand norm3.123286;
object readout/decoded-object gradients exactly absent, as intended. Calibrated
weight0.05 (bounded cap), initial geometry/base gradient ratio0.43191, ramp100.
Four-TRAIN GT-tau geometric contact MAE0.01908; observed predicted-tau MAE0.22583.
Proxy/transfer mismatch is measured rather than assumed zero.

Actual capacity96 completed8 full optimizer updates, peak15.486GiB allocated/
17.203GiB reserved,0.3691updates/s. Main local96/global288 selected unchanged.
Real3-rank readiness8 saved and audited; strict2 recovery saved10 and audited:
all568 optimizer states at8/10,3 RNG payloads, exact scales, all tensors finite,
equal3-rank signatures. Readiness peak15.859GiB allocated/17.639GiB reserved;
recovery15.458GiB allocated/17.789GiB reserved. Sustained training resumes10.

Physical initialization parity passes against the original strict-FP32 arm,
all available DEV sources' hand and object-h24 differences<0.0001mm. GRAB init
hand52.68717mm/object-h24=15.82866mm; new geometry contact MAE0.14122,
near-contact MAE0.30446, GT-tau proxy MAE0.01423, coverage100%.
New and old total objectives differ by design; compare physical metrics and
same-definition C metrics, not raw total loss across objective families.

Live startup at60: loss0.729200, pre-clip norm0.9160,
shared/context RMS3.860/5.877,
peak allocated16.012GiB, all finite. Warmups are not yet complete.
Initial throughput~0.32updates/s, target10000 would take~9h before additional
periodic overhead, so the4h bound likely binds first. Free178.56GiB;
startup receipt/config delta/init-parity receipts retained in output root.
No efficacy or rollout-drift claim from this short startup.

### Sustained acceptance at60

LR warmup50 passed. Loss0.729200, pre-clip norm0.9160,
shared/context RMS3.860/5.877,
peak allocated16.012GiB. Geometry weight is still on its
100-update ramp; no full-ramp or later-efficacy claim. Latest selected GPUs
sampled100%, physical22269/21117/20831MiB, free178.56GiB.
Progress-window rate0.245updates/s includes initial DEV/save
overhead; preserve the4h cap. startup-acceptance.json refreshed with exact
preflight, stage audits, full config delta and current telemetry.
