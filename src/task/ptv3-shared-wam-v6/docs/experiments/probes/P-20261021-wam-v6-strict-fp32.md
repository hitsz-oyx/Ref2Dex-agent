---
schema: ref2dex.probe.v2
probe_id: P-20261021-wam-v6-strict-fp32
experiment_id: P-20261021-wam-v6-strict-fp32
date: 2026-10-21
task: ptv3-shared-wam-v6
branch: main
git_commit: 1b57a487a8309b6223983fd34019d8178d9e122c
claim_id: C1
hypothesis_family: HF-v6-training-precision
probe_index_in_family: 1
seed_pool: probe
seeds: [170,171]
decision_changed_if_positive: retain strict FP32 if activation scale and physical-unit DEV errors remain better than the matched BF16 ancestor run
decision_changed_if_negative: investigate optimizer-residual conditioning rather than assuming precision alone fixes drift
status: RUNNING
run_id: strict-fp32-pretrain-20261021-r1
---

# Healthy-checkpoint strict FP32 geometry training

Decision Probe explicitly requested by the user. Distinguish numerical-precision
sensitivity during training from an instability that remains in FP32. Cheapest
useful test: preserve the existing geometry objective, healthy source4371,
expanded local TRAIN/DEV recipe, seed streams and three-GPU batch; change
arithmetic precision, while monitoring activation amplitude and physical errors.
Existing degraded weights still perform poorly under strict FP32 inference;
that does not answer whether healthy-start FP32 training prevents the instability.

Result:38 tests and actual local96 capacity / world3 save8 / strict recovery2 PASS; sustained FP32 training RUNNING.
Decision: run bounded healthy-start FP32 training; do not infer a cure from startup.

## Fixed protocol and environment

- Source: outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/training/step004371.pt.
  Model weights and all4 horizon-normalizer buffers copied exactly. Fresh AdamW,
  RNG/warmup/count/selection, matching the previous geometry run's initialization
  protocol; this is not optimizer continuation from the no-geometry objective.
- Matched recipe: outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json
  and stats.json;207579 TRAIN/3356 DEV windows,1177 cached sequences, GRAB dominant.
  No new raw data/cache or normalizer fit. Retain goal endpoint and event plans,
  H4/K24/joint21,768-wide12-layer shared stack and existing PTv3 architecture.
- Baseline configuration: outputs/ptv3-shared-wam-v6/geometry-consistency-20261021-r1/config-b96.json.
  LR1e-5, warmup50, global clipping1, geometry weight0.022920552754813268,
  geometry warmup100 and clearance weight0.25 unchanged; no new calibration.
- Strict precision: amp=false, tf32=false, strict_fp32=true. Per-model task-owned
  adapters replace PTv3's forced BF16 FlashAttention and dense MHA with float32
  PyTorch memory-efficient SDPA; no half fallback, no vendor modification and no
  parameter/state-key changes. Matmul, cuDNN and spconv TF32 disabled. Inference
  Transformer fastpath disabled so DEV uses the same adapter. All model
  parameters float32; runtime QKV dtype assertions. Cached surface offsets and
  contact labels remain in their original storage dtype: this tests computation,
  not a new data-precision preprocessing pipeline.
- User explicitly retains previously authorized GPUs4/5/6, three RTX3090 cards.
  Unrelated0–3/7 untouched. /home2/wyy/miniconda3/envs/graspenv/bin/python,
  Python3.8.20/Torch2.4.1+cu121/CUDA12.1,8 spawn loaders/rank,
  OMP/MKL2, OPENBLAS1, NCCL_P2P_DISABLE=1. Actual versions in run manifest.
- Capacity:8 disposable full optimizer updates on GPU4 at local96; only lower
  to80/64 on actual OOM or reserved peak>21.5GiB. If lower, record the changed
  effective global batch and limit causal interpretation. Main weights always
  reinitialize from4371, never from the disposable capacity trajectory.
- Actual3-rank readiness8 updates/save, strict recovery2 updates/save, then
  sustained training from10. Audit568 finite optimizer states, exact scales,
  RNG3/world3/effective batch and rank signatures at save/recovery boundaries.
- Bound:10000 new-objective updates or14400 cumulative training seconds,
  whichever first. DEV before updates,20 then every500, fixed seed171 and32
  examples/source/microbatch4. Preserve physical hand/scene and geometry metrics.
  Track existing-forward shared/context RMS on logged TRAIN steps; these random
  minibatch values are distinct from fixed-panel diagnostic RMS measurements.
- Storage: stage checkpoints<=8GiB; combined expected peak<20GiB; initial free
  about193GiB. Stop at20GiB free, above user's15GiB hard floor. Root STOP requests
  saved shutdown; no-progress20min/stage deadlines enforced. Preserve all earlier
  source/checkpoint/results. No cleanup or data migration needed for this task.

Outputs: outputs/ptv3-shared-wam-v6/strict-fp32-pretrain-20261021-r1/.
Frozen tracked code and launch log: tmp/strict-fp32-pretrain-20261021-r1/.
Entrypoint: tools/run/launch_strict_fp32_training.py within this Task.

## Engineering verification

38 v6 tests pass in9.81s, including7 precision tests. Ragged FP32 forward/backward
matches an independent FP64 segmented reference; efficient CUDA FP32 agrees
with math FP32; dense self/cross attention agrees with native MHA outputs and
parameter/input gradients with bool/float padding masks and unaligned sequence
length10. State tensors unchanged; legacy configuration remains opt-out.

## Limitations and interpretation

This is a matched exploratory precision arm, not an independent replication or
validation. Efficient FP32 and BF16 Flash kernels differ, so a gain attributes
to this precision/backend treatment together. Healthy source itself was trained
with BF16/TF32; no recovery of lost data precision is implied. Constant data/LR/
losses do not guarantee stability, and contact geometry remains an approximate
proxy. Judge effectiveness from later fixed DEV and identical rollout panels,
not one noisy loss or short startup. Four-hour budget may bind before10000.

## Actual launch and acceptance boundaries

Frozen code1b57a487, supervisor1234454. Healthy source4371 audit PASS:
world3/global288,568 finite Adam states all4371,3 RNG entries and exact scales.
Initialization itself copies model/normalizers only and resets the optimizer,
as declared above. Capacity96 completed8 real full optimizer updates;15.433GiB
peak allocated/17.156GiB reserved,0.4007 updates/s; no capacity checkpoint.

Actual3-card readiness8 completed, then2 strict recovery updates completed10.
Both checkpoints audited PASS,568 finite states all8/all10,3 RNG entries,
world3/global288 and exact normalizer buffers. All3 parameter signatures equal
at both boundaries. Readiness allocated15.80GiB/reserved18.03GiB; strict recovery
reserved17.74GiB. Sustained training loads step10 with original Adam/RNG/count,
not another restart. Actual run manifests record16 PTv3 serialized and13 dense
attention adapters, float32, all TF32 modes disabled.

An additional real-head-shape FP64 differential checks efficient FP32 at
12×64/330,8×32/330 and16×16/128: max forward2.2814e-6, input gradient1.9561e-6,
relative L2<7.71e-7. Float32 atol/rtol1e-5 checks PASS; an initial absolute-only
2e-6 threshold was too tight at the8×32 shape and is recorded in the receipt.
BF16-rounded-input reference errors0.00326–0.01404, clearly distinct from FP32
rounding. Receipt: fp32-kernel-reference.json in the output root. This is a
numerical engineering check, not evidence that training instability is fixed.

Fixed GRAB DEV comparison (mm, same inherited seed171 panel):

| Arm / update | hand | hand h24 | object h24 | geometry relative |
| --- | ---: | ---: | ---: | ---: |
| BF16 geometry /0 |52.6763|102.5091|15.8196|22.5379|
| FP32 geometry /0 |52.6872|102.5778|15.8287|22.5406|
| BF16 geometry /20 |52.4849|101.9785|15.2661|22.4074|
| FP32 geometry /20 |52.5363|102.1186|15.0312|22.4491|

These early results establish comparable initialization and finite updates;
they do not demonstrate a persistent advantage or improved rolling execution.

## Sustained startup acceptance at60

Actual FP32 training has passed LR warmup50 and reached60; loss0.726686,
pre-clip norm0.9180, shared/context RMS
3.863/5.887,
all finite. The geometry coefficient is still on its declared100-update ramp.
Recent40→60 throughput0.329 updates/s (~94.8 samples/s), slower than
BF16 as expected; initial compute-only0.36 updates/s excludes some overhead.
At this rate10000 would need roughly8h, so the4h time budget likely binds first.
Do not advertise10000 as an expected completed target within4h.

Latest6 supervisor samples average GPU4/5/6 utilization96.5/96.3/97.3%, physical
memory22225/20733/21125MiB; main allocated peak15.84GiB at this point.
These utilization values include synchronization; they do not independently
measure useful FLOPs. Free disk187.09GiB, well above20GiB stop boundary.
Startup acceptance receipt records capacity,8+2 audits, precise backend/runtime,
progress/telemetry and bounds at outputs/ptv3-shared-wam-v6/
strict-fp32-pretrain-20261021-r1/startup-acceptance.json.

Decision: continue the existing bounded FP32 run. Current finite activations
are engineering acceptance, not proof that later FFN amplification or rollout
drift is fixed. Compare fixed DEV500/1000+ and the existing frozen rollout panel
before promoting a numerical-stability or tau-accuracy claim. No further
training architecture/loss/LR changes were introduced for this precision arm.

### Full-weight100-update acceptance

Sustained training now reached100; both LR and geometry warmups
complete. Loss0.758336, pre-clip norm0.8510,
shared/context RMS3.866/6.060,
peak allocated15.966GiB. Recent60→100
throughput0.303updates/s. Free187.08GiB. No amplitude explosion yet in this
short prefix; later stability and rolling accuracy still unproven. Main acceptance
receipt refreshed; original60 snapshot retained as startup-acceptance-step60.json.

## Fixed DEV500 / live620 monitoring

Same fixed GRAB DEV panel, mm:

| Metric | FP32 init | FP32 step500 | BF16 step500 |
| --- | ---: | ---: | ---: |
| Hand mean |52.6872|52.9030|53.0297|
| Object h08 |30.9338|32.0628|32.2785|
| Object h24 |15.8287|13.2844|16.7900|
| Geometry relative |22.5406|21.0771|21.2564|

Live620 loss0.72510, pre-clip norm0.9040, shared/context
RMS3.894/6.009,
finite. Latest throughput0.252updates/s; GPUs4/5/6 sampled100%,
free disk185.90GiB. Continue unchanged under the existing4h bound.
Object h24 improves versus own initialization and matched BF16 step500; hand
and object h08 still do not improve versus initialization. BF16 also had finite
small gradients through600, so this prefix cannot show prevention of its later
instability. No new rolling execution/video evaluated. Receipt in output root:
status-dev500-comparison.json. Next decision checkpoint remains DEV1000+.

## Fixed DEV3000 / live3060 monitoring

The precision/backend arm has crossed the old BF16 instability interval. Logged
FP32 max gradient through3040=1.4248 versus BF16=237.22; at3040 norms0.7614 versus
83.9506. Shared/context RMS remain3.634/5.640 in the live FP32 batch. These
minibatch RMS measurements are not identical-input checkpoint diagnostics.

Same fixed GRAB DEV, mm:

| Metric | FP32 init | FP32 step3000 | BF16 step3000 |
| --- | ---: | ---: | ---: |
| Hand mean |52.6872|52.2290|56.6400|
| Object h08 |30.9338|33.3978|34.3479|
| Object h24 |15.8287|14.1503|39.4621|
| Geometry relative |22.5406|19.2128|27.1794|

Object h24 is about64% lower than matched BF16, and~11% lower than own init.
Hand mean changes little versus init; the consumed8-frame object endpoint still
worsens~8% versus init. Available-source macro hand32.6466→33.2437mm and total
DEV objective1.12533→1.17428 also worsen; do not promote GRAB results to an overall
accuracy improvement. Macro object-h24=9.8422→9.6464mm (BF16 step3000=31.4737).
No new rolling execution/video measured. Precision and kernel backend changed
together, so this one matched exploratory arm does not isolate precision alone.

Actual run continues with no config changes; loss0.69814,
throughput0.270updates/s, allocated peak15.988GiB,
physical~22GiB/card, GPUs99–100%, free184.27GiB. Remaining declared
training budget approximately48.8min; keep original
4h boundary and saved stop. Receipt: status-dev3000-comparison.json in run root.
Decision: finish the bounded arm; stability has improved through3000, but useful
8-frame trajectory gain and rolling drift reduction remain unresolved.

## Decision diagnostic: fixed TRAIN versus DEV gap

Question: has optimization itself saturated, or does TRAIN improve while unseen
sequences regress? Before changing the continuing training, evaluate32 uniform
GRAB TRAIN indices (seed170, fixed across checkpoints) at healthy4371 and strict
FP32 step3000, using the same real evaluation function and current strict FP32
backend. Existing fixed DEV seed171 panel supplies the generalization side.
Cheapest distinguishing check:64 examples total, inference only, microbatch2;
no optimizer steps or running-config change. If TRAIN physical/loss metrics
improve while DEV worsens, prioritize generalization/data balance/early selection;
if both fail on the consumed prefix, next distinguish objective-gradient tradeoff
from optimization underfit. This alone does not causally establish loss conflict.

Resources: GPU7 shared only after confirming monica's owner, utilization0% and
16.15GiB free; bounded180s/32 examples per checkpoint, expected<3GiB new peak,
no writes to external data. Running4/5/6 unchanged;20GiB storage floor preserved.
Python graspenv/Torch2.4.1+cu121, model and precision code1b57a487, eval mode and
real current-observation/planned-event data. Seed170 is diagnostic reuse from
the same exploratory arm, not an independent replication. Audit script commit
and selected indices will be saved in the report under the existing run.

### Fixed TRAIN diagnostic completed: generalization gap, not demonstrated capacity saturation

Bounded probe completed in20.10s on GPU7, peak allocated<0.453GiB.
Audit commit9488e449,32 fixed GRAB TRAIN indices/seed170, source4371 and FP32
step3000, strict FP32/eval and unchanged production evaluator; no optimizer or
main training changes. Initial r1 stopped at source-pool key lookup before
inference (integer1 versus reader's string'1'); corrected r2 is the completed
report. Source/data/checkpoints remained intact; failed log retained in project tmp.

| Metric / mm | Fixed TRAIN init | Fixed TRAIN3000 | Fixed DEV init | Fixed DEV3000 |
| --- | ---: | ---: | ---: | ---: |
| Hand mean |23.8091|21.1200|52.6872|52.2290|
| Hand prefix1–8 mean |8.6147|8.1009|15.6812|15.8990|
| Object prefix1–8 mean |6.5086|5.7843|17.0177|18.1306|
| Object h08 |10.9751|9.3245|30.9338|33.3978|
| Object h24 |16.7872|15.5813|15.8287|14.1503|
| Geometry relative |17.9920|14.0825|22.5406|19.2128|

TRAIN scene normalized loss0.06342→0.04653, hand0.04269→0.03453, full
objective0.79342→0.72669. This is a same-input checkpoint differential, stronger
than comparing unrelated random training batches. Fixed DEV scene normalized
loss instead0.31926→0.34456; GRAB contact BCE roughly flat0.30937→0.30867,
part CE worsens0.53025→0.55057. All-source macro objective rises. Current draws
are99.809% GRAB, explaining a plausible cross-source adaptation pressure; the
within-GRAB gap means source imbalance alone is not an established full cause.
Only32 examples/panel; no uncertainty interval or universal overfitting claim.

Read-only code checks explain why the metrics can diverge, without proving a
sole causal mechanism: V5 sample_window supplies goal_object_delta=future_flow[-1],
i.e. the offline24-frame object endpoint is a conditioning input. Intermediate
motion still must be inferred. The training objective combines24-horizon
normalized squared coordinate error, contact BCE, part CE and relative geometry;
it is not the consumed1–8-frame physical EPE metric. Coordinate normalizers
already vary by horizon, so do not claim all horizons have equal physical weights
or that simply upweighting early frames is a validated remedy. Geometry's common
translation invariance is covered by the existing test; relative contact can
remain consistent while both trajectories share an absolute-position error.
No gradient-cosine conflict measurement or architecture-capacity ablation has
been run, so neither loss conflict nor FFN capacity exhaustion is established.

Decision: keep the bounded precision arm unchanged to its declared boundary.
Do not extend training or enlarge FFN merely because TRAIN still improves.
Next discriminating work should target held-out sequence/object/action slices
and a frozen short-prefix checkpoint criterion; any data balancing/loss-weight
change needs a separate matched arm. Rolling evaluation remains required before
claiming drift reduction. Evidence: fixed-train-gap-r2/report.json.
