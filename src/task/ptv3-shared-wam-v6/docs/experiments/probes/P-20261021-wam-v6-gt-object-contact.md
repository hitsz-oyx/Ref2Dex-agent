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
status: STOPPED
run_id: gt-object-contact-20261021-r1
---

# GT2048 object and predicted-hand contact probability supervision

Decision Probe requested and clarified by the user: stop the current FP32 run,
then use complete2048-point GT future object geometry with predicted tau to
compute contact probability and align it to GT contact labels. User explicitly
selected this over the old relative-vector loss, whose object terms would cancel
if both sides used GT object geometry. This is a new objective, not strict resume.

Result: new arm saved/stopped3238 at its4h boundary; final checkpoint audit PASS; fixed DEV3000 shows no short-prefix or GT-contact gain.
Decision: do not extend this GT-object probability package; preserve checkpoints and investigate the held-out trajectory gap before another training arm.

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

## DEV500 / live640: no demonstrated early gain

Run remains RUNNING, loss0.745052, norm0.9311, shared/context RMS
3.820/5.925; no scale explosion.
Selected GPUs sampled100%, throughput0.226updates/s, free177.40GiB.
Same fixed GRAB DEV comparison, mm except contact probability:

| Metric | Init | Previous FP32 step500 | GT-object step500 |
| --- | ---: | ---: | ---: |
| Hand mean |52.6872|52.9030|53.2639|
| Hand prefix1–8 mean |15.6812|15.6216|15.6837|
| Object prefix1–8 mean |17.0177|17.4378|17.4950|
| Object h24 |15.8287|13.2844|13.5366|
| Predicted-both geometry relative |22.5406|21.0771|22.6323|

GT-object geometric contact MAE0.141219→0.141255, effectively flat;
near-contact MAE0.304455→0.302464, small change only. Available-source macro
hand32.6466→32.8360mm (previous FP32 step500=32.6810mm), no macro gain.
The favorable object-h24 change relative to init also occurs in the previous
arm, so it does not establish a new GT-object-contact benefit. No new rollout.
Continue unchanged within the already declared4h bound, reassess fixed DEV1000;
this early comparison does not establish eventual failure or which changed
ingredient caused the result. Receipt: status-dev500-comparison.json.

## DEV3000 / live3040: no held-out benefit through the bounded arm

Stable optimization persists: loss0.686413, norm0.7491, shared/context
RMS3.618/5.590.
Same fixed GRAB panel, mm:

| Metric | Init | Previous FP32 step3000 | GT-object step3000 |
| --- | ---: | ---: | ---: |
| Hand mean |52.6872|52.2290|52.5880|
| Hand prefix1–8 mean |15.6812|15.8990|15.9814|
| Object prefix1–8 mean |17.0177|18.1306|18.2310|
| Object h24 |15.8287|14.1503|14.6603|
| Predicted-both geometry relative |22.5406|19.2128|21.7026|

GT-geometry contact MAE0.141219→0.141773 and near-contact
0.304455→0.305500: neither improves. Available-source macro hand
32.6466→33.3826mm (previous arm33.2437mm). Most new-versus-old physical
changes are small; this32-example exploratory panel supplies no evidence of
benefit, not a statistically established universal degradation. No rollout run.
Random TRAIN-window means decline (see receipt), so finite optimization alone
has not produced held-out contact/prefix improvements. No claim that object
noise is irrelevant: this package also changes loss form, query count and weight.

Decision: do not extend the experiment. Leave the existing saved-stop4h boundary
in place; remaining approximately13.4min at this snapshot.
Current status remains RUNNING; it has not stopped yet. Receipt in run root:
status-dev3000-comparison.json. Further spending on this exact package is not
supported by the observed fixed-panel results.

## 2026-10-22 Final saved stop and repository result checkpoint

Actual execution status STOPPED normally at3238, exit code0; no loader error.
This sustained stage performed3228 updates following the real8+2 readiness/
recovery chain, so the new-objective total is3238. The final complete checkpoint
is training/step003238.pt,1,186,351,949bytes. Independent CPU audit PASS:
568 finite Adam states all3238, model tensors finite,3 RNG payloads,
world3/local96/global288, and all4 normalizer buffers exactly equal to inherited
TRAIN stats. Supervisor1572421 and torchrun1583025 exited. No restart or budget
extension. Audit receipt: final-stop-audit.json in the output root.

The final checkpoint cumulative new-objective elapsed time is14402.59s; the
sustained-stage result alone reports14286.84s, excluding inherited readiness/
recovery time. The small overrun is one update/save boundary, not a new budget.
Supervisor wall time14555.96s also includes disposable capacity/preflight and
startup. Peak allocated16.040GiB/reserved21.285GiB, recent compute throughput
0.2397updates/s. Preserved main checkpoints:20,3000,3238, plus separate
readiness/recovery artifacts; no checkpoint overwritten.

Latest actual fixed DEV remains3000. There is no DEV3238 evaluation, so the
reported hand52.588/object-prefix8=18.231/object-h24=14.660mm and GT contact
MAE0.141773 are the3000 checkpoint results, not final3238 metrics. This package
has not demonstrated benefit over the earlier strict-FP32 arm; do not attribute
the result solely to GT object source or claim statistical universal harm.
No rolling evaluation or robot success measurement was run for this arm.
Implementation and prior44-test/real-GPU checks were already committed; this
repository checkpoint finalizes status, evidence links and the bounded decision.
