---
schema: ref2dex.probe.v2
probe_id: P-20261021-wam-v6-geometry-consistency
experiment_id: P-20261021-wam-v6-geometry-consistency
date: 2026-10-21
task: ptv3-shared-wam-v6
branch: main
git_commit: b852f39cb016b2a04332c3ffd5a09037cc3ca7db
claim_id: C1
hypothesis_family: HF-v6-geometry-consistency
probe_index_in_family: 1
seed_pool: probe
seeds: [170,171]
decision_changed_if_positive: retain the geometric objective for fixed-sequence rollout comparison while monitoring physical-unit trajectory errors
decision_changed_if_negative: revisit geometry correspondence quality or gradient balance before expanding this method
status: COMPLETED
run_id: geometry-consistency-20261021-r1
---

# Predicted hand/object contact geometry consistency

Decision Probe explicitly requested by the user: implement geometric consistency
and initialize from the current WAM checkpoint. Distinguish whether a direct
gradient to both coordinate readouts reduces relative hand/object contact error
without degrading held-out tau/object predictions. The cheapest discriminating
setup adds a deterministic loss branch to the existing v6 model, keeps the
expanded local dataset/goals/event inputs and uses the latest complete baseline
checkpoint. No new independently trainable contact decoder or future-hand input.

## Geometry and objective

The previous C head predicts labels from shared features, with no gradient to
the final hand_future/scene_future coordinate readouts. This branch computes
geometry directly from their denormalized outputs:

- Bind each observed778-point hand skin to its nearest observed joint21 bone,
  with a palm-normal-referenced local frame and projected bone coordinate.
  Predicted joints transport that skin differentiably. Attachments read only
  observed input; GT futures enter only the loss as targets.
- Extend the predicted32-point object flow to selected CAD surface queries with
  an observed-only affine pseudoinverse. This reproduces rigid GT flow for
  full-rank CAD samples; it is not a rigidity projection of arbitrary predictions.
- At frames1/4/8/12/16/20/24, select64 highest GT-proximity and64 uniformly spaced
  queries from2048. The original C supervision still covers all24×2048 outputs.
  GT positive queries (>0.25) choose a nearest skin vertex of the GT-labelled
  anatomical part. Compare the predicted object-minus-skin vector to the vector
  obtained from the same proxy under GT trajectories. This calibration makes
  exact GT predictions zero-loss despite approximate skinning; no forcing of
  joint centres onto object surfaces or zero contact-probability collapse.
- Positives use SmoothL1(relative-vector error/10mm), with no far-distance
  cutoff on predicted gaps, so a separated predicted hand receives gradients.
  Proxy GT gaps>50mm or unavailable hand/part geometry are masked and coverage
  is reported. Unknown geometry is never fabricated as a negative.
- GT-negative queries (<0.01) receive only local clearance penalties when the
  predicted nearest skin is too close. Margin=min(GT-proxy gap,33mm)-2mm;
  detached GT direction supplies a separating gradient at exact overlap.
  Distant noncontact hands are not pulled inward. This is not a full signed
  mesh collision, self-contact, force or stability guarantee.

Total loss is original tau/scene/C/part plus
geometry_weight × (geometry_contact +0.25 geometry_clearance). A four-example
TRAIN-only GPU preflight measures gradients at actual hand/object output
readouts. Choose weight=min(0.05,max(0.0001,0.5×base_gradient/geometry_gradient)),
then freeze it and ramp it over100 updates. Require positive proxy coverage>=80%,
finite nonzero direct gradients to both decoded outputs/readouts, and no direct
readout gradient from the original C-only loss. This calibrates scale, not an
efficacy result or an independent validation.

## Initialization, resources and budget

Keep the baseline running during implementation/tests. Then request owned STOP
on local-expanded-ddp-20261021-r1, wait for saved exit, and load its newest actual
final checkpoint. Copy model weights and all normalizer buffers exactly. Fresh
optimizer/RNG/count/warmup/selection for the new objective, explicitly recorded
as weights-only initialization; do not claim strict old-objective continuation.
Initialization step/path is written in the actual run manifest and switch receipt.

Reuse TRAIN207579/DEV3356 windows over1177 sequences; no input/cache mutation.
Goal endpoint and v6 CONTACT/RELEASE event conditioning retained. Three local
GPUs4/5/6 remain authorized by the user's previous multi-GPU instruction; do not
touch unrelated0–3/7 work. Python graspenv/Torch2.4.1+cu121, BF16/TF32,
8 spawn loaders/rank, NCCL_P2P_DISABLE=1. Preserve LR1e-5,50-update LR warmup.

Capacity test8 disposable updates at local96, falling back80/64 only if OOM or
reserved peak>21.5GiB. Then initialize8 real three-rank updates, audit checkpoint,
strict resume2, audit again, continue the same new-objective checkpoint. Three
parameter signatures and all568 optimizer states/RNG are checked. No capacity
updates are consumed by main training.

Fresh bounded run:10000 updates or14400 cumulative new-objective seconds,
whichever first; DEV before any update and every500, initial step20,32 fixed
samples/source, microbatch4. Keep original physical-unit hand/scene/C metrics
and add geometry relative EPE, predicted/GT proxy gaps, negative violations and
positive coverage. Compare physical metrics against this exact initialization;
new total objective is not numerically comparable to the old objective.
Per-stage8GiB checkpoint budget; transition/readiness/training combined~16GiB
peak reserved; stop at20GiB free, always exceeding user's15GiB hard floor.
Latest baseline and previous results remain untouched in their original run.

Outputs: outputs/ptv3-shared-wam-v6/geometry-consistency-20261021-r1/.
Code archive/logs: tmp/geometry-consistency-20261021-r1/.

## Engineering evidence

7 new geometry tests pass: rigid skin/flow equivariance, GT-zero and common-motion
invariance, far-contact gradients to both outputs, noncontact no-pull/overlap
separation, missing-geometry/NaN masking, gradients to actual SharedWAMV6
coordinate readouts, and explicit weights-only initialization/normalizer binding.
The final complete v6 suite passes30 tests in8.62s under graspenv. GPU preflight
and the actual three-rank save/recovery checks also pass, as recorded below.

Method limitations: skinned observed surface is an approximate future mesh,
and arbitrary predicted object flow is extended affinely. Paired-GT calibration
avoids a known proxy target bias but does not prove exact contact or prevent
every penetration. No matched no-geometry continuation control or physical
rollout comparison has yet run; no claim of solving drift or improving grasping.


## Actual launch and startup evidence

Frozen implementation b852f39cb016b2a04332c3ffd5a09037cc3ca7db, supervisor943417,
run geometry-consistency-20261021-r1. The old expanded run stopped normally at
4371; its complete1,186,352,333byte training/step004371.pt is the initialization,
not an earlier best checkpoint or remote23758. CPU audit passes finite model,
568 Adam states at4371,3 RNG payloads, world3/global288 and inherited normalizers.
All old model parameters/buffers are copied; optimizer/RNG/step/selection are
explicitly reset for this new objective. Later8→10 recovery is strict within
this new objective.

Actual current-checkpoint GPU preflight on four TRAIN examples passes. Original
C/part-only gradients to both coordinate readouts are None. New geometry gives
hand_readout16.60445/object_readout20.80343 and decoded hand5.33547/object1.60874
(nonzero finite gradient norms). Eligible positive coverage2897/2897=100%.
Base coordinate-readout norm1.220175 versus raw geometry26.617487 selects frozen
weight0.022920552754813268, yielding full-weight gradient ratio0.5; ramp100 updates.
These four TRAIN examples only calibrate the loss and check connectivity.

Actual batch96 capacity8 updates passes, allocated13.594GiB/reserved15.342GiB.
Keep three3090 GPUs4/5/6, local96/global288. Real readiness8 updates saves a
complete1,186,348,749byte checkpoint:568 optimizer states at8,3 RNG payloads,
identical three-rank parameter signatures and exact inherited normalizers.
Strict recovery8→10 passes the same audit (1,186,351,629bytes, all states10).
Sustained training resumes10 and has reached100 with finite loss0.75918,
grad_norm0.86805, nonzero geometry_contact0.55237/geometry_clearance0.07497 and
full weight0.02292055; allocated peak14.125GiB. Instantaneous selected GPU
utilization100/95/99%, physical memory21,013/20,299/19,001MiB; free197.71GiB.
No OOM, NaN, loader failure or unrelated process termination observed.

The fixed32-GRAB-example DEV panel at the exact4371 initialization has hand
EPE52.67626mm, geometry relative22.53791mm, predicted proxy gap23.53811mm,
GT-proxy gap4.22592mm, noncontact violation2.06307mm, positive coverage100%.
This is the correct before point for later geometry-run comparisons; the
historical remote23758/local-original49.9303mm panel starts from different weights.
DEV at20 is operational startup evidence, not a meaningful efficacy verdict.

Full trace: process.json, gpu-preflight/report.json, config-b96.json,
readiness/dev-step000000.json, per-stage result/checkpoint audits, training/
progress.json and train.jsonl under the declared output. Source/runtime metadata
are in each run-manifest; code archive and supervisor log are project-local tmp.
Root STOP requests a saved stop. Enforced budget10000 updates or cumulative4h,
20GiB free-space floor, bounded checkpoint retention; source checkpoint retained.
No free-feedback rollout or physical grasping evaluation yet.

Startup acceptance: startup-acceptance.json records the exact40→100 update
interval (240.56 samples/s), six recent GPU samples, full gradients/audits and
exact initialization DEV. Compute-only remaining ETA3.29h excludes periodic
DEV/save overhead; cumulative4h bound remains authoritative.


## Read-only gradient diagnosis after sustained degradation, 2026-10-21

User asks why object supervision does not prevent drift. The scene objective
is present with scene_weight=1: masked motion-weighted normalized point-flow
MSE, alongside hand_weight=1 and C/part. Actual backward includes this objective;
clip_grad=1 applies to all parameters before AdamW.step. A relative contact
objective alone is invariant to common translations, but the scene and hand
supervision break that invariance. This geometric ambiguity cannot by itself
explain failure to minimize the full supervised objective.

Deterministic four-example TRAIN audit uses identical indices109941/121397/
156119/158552, seed170, train-mode BF16, source4371 versus pinned geometry4500
versus saved geometry7500. No optimizer updates. Output:
outputs/ptv3-shared-wam-v6/geometry-gradient-diagnosis-20261021-r1/report.json.
Entry: tools/audit/diagnose_geometry_balance.py. Real CUDA execution completed
three checkpoints in16.45s, peak allocated1.944GiB on GPU7, leaving the active
three-card run and other processes untouched.

| Checkpoint | Scene loss | Original loss gradient norm | Scene-only norm | Weighted geometry norm | Scene/geometry cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| source4371 | 0.03406 | 5.354 | 2.765 | 1.019 | 0.171 |
| geometry4500 | 0.38209 | 2455.665 | 2432.540 | 326.372 | 0.904 |
| geometry7500 | 4.72008 | 163.368 | 157.655 | 6.751 | 0.711 |

Weighted geometry loss0.01625/0.04364/0.11181, original objective0.88714/
1.79343/6.06263. Thus this fixed batch does not support the simple hypothesis
that a dominating geometry gradient opposes/suppresses object supervision:
removing the geometry graph at4500 still leaves original gradient norm2456,
mostly from the scene objective, and scene/geometry gradients are aligned.
This does not establish the historical trigger or prove geometry harmless on
other batches. Full matched continuation is still pending, not replaced by
this same-checkpoint gradient deletion.

TRAIN log median preclip norm100–500 is0.936;1000–2000 is11.85;2000–3000 is57.93;
3000–4000 is200.88;4000–4500 is818.90. Scene loss itself rises, then increases
abruptly around5140. The symptom is sustained optimization instability on TRAIN
as well as DEV, rather than only worse generalization or a smaller valid
relative-contact error. At4500 the audited total norm2749.26 implies global
clip factor0.000364; Adam preconditioning/history means this is not itself a
measurement of the actual parameter update or a proven cause. Next diagnostic
should localize scene-readout sensitivity and inspect optimizer updates, with a
matched no-geometry continuation to distinguish triggering mechanisms. No fix,
new long training, stop, or active-config mutation was performed by this audit.


## Layer, precision and optimizer localization (saved diagnostic checkpoint)

Decision: distinguish a geometry-derivative singularity, attention numerical
amplification, shared-feature scale drift, and optimizer corruption before
choosing a repair. Reuse the same four actual TRAIN examples and preserved
4371/500/4500/7500 checkpoints. All source files and active training remain
unchanged. Single-batch GPU7 diagnostics, no new production training, no saved
model/checkpoint. Two standalone audit tools are now versioned:
`tools/audit/diagnose_scene_readout.py` and
`tools/audit/diagnose_optimizer_update.py`.

Forward hooks localize the scale increase upstream of the scene readout:

| Activation RMS | Source4371 | Geometry4500 |
| --- | ---: | ---: |
| PTv3 scene tokens | 1.62 | 1.86 |
| Shared Transformer layer0 | 1.99 | 2.31 |
| Shared Transformer layer3 | 2.51 | 5.51 |
| Shared Transformer layer5 | 2.81 | 36.47 |
| Shared Transformer layer11 | 3.58 | 423.15 |
| Scene context projection | 5.58 | 3408.50 |

Input tensors and goals are identical across checkpoints. The main scale drift
appears across the shared pre-norm Transformer residual stream; PTv3 features
remain near their original scale on this batch. The shared encoder has no final
output LayerNorm; scene_context consumes its raw residual stream. This
localizes an amplification path, not yet the original optimization trigger.
At4500 projected attention logits reach approximately7.54e6. The isolated scene
head reproduces large gradients with the upstream activations detached and
without any contact/geometric loss, so a hand-skin derivative is not necessary
to reproduce this failure at the degraded checkpoint.

Same captured activations, no weight update: BF16, mathematical SDP, FP32,
FP32 attention alone, and explicit FP64 attention reference were compared.
A representative4500 capture gives scene-head gradient norms1992.15 in BF16,
1193.76 in FP32 and1167.70 with FP64 attention. Query-input gradient norms are
56.40/0.01762/0.00724 respectively; thus low precision amplifies a sensitive
path, but higher precision alone leaves a genuinely ill-conditioned readout.
Centering the keys (mathematically softmax-shift invariant) gives1176.47 in
FP32 and does not remove the large parameter gradients either.

Stateless shared-output LayerNorm lowers a4500 isolated-head gradient from
about1127 to25.8; context LayerNorm lowers it to9.73. However the scene loss
increases0.375→3.295/3.740. This is a counterfactual localization, not a validated
repair: inserting normalization into the already trained checkpoint changes
its function and cannot be hot-applied as if it preserved predictions.

Six disposable one-step checks (500/4500 × restored Adam/fresh Adam/SGD), no
checkpoint write. At4500 restored Adam changes same-batch total1.83707→1.80021;
fresh Adam at full1e-5 changes1.83707→3.30338; clipped SGD changes it to1.80654.
This does not prove stored optimizer corruption or that resetting Adam fixes
anything. Four samples and a full-LR reset differ from the actual world3/
global288 training and50-step warmup; no matched-continuation conclusion.
Stored Adam can improve this batch despite a locally positive gradient/update
dot product, emphasizing that BF16/conditioning and finite steps prevent a
simple first-order sign argument from resolving the trigger.

Audit reports live under outputs/ptv3-shared-wam-v6/:
scene-readout-diagnosis-20261021-r1/ (12.36s),
scene-activation-diagnosis-20261021-r1/ (11.41s),
scene-normalization-diagnosis-20261021-r1/ (10.31s),
scene-key-centering-diagnosis-20261021-r1/ (8.74s), and
optimizer-update-diagnosis-20261021-r1/ (24.84s).
Peak one-step CUDA allocation2.14GiB, within available GPU7 capacity.
Each has report.json with arguments/indices/versions, per-mode/per-step losses,
gradients, activation scales and memory. Different seeded full-model captures
show small CUDA floating-point variations that the ill-conditioned readout
magnifies; within each precision comparison the captured inputs are identical.
Do not treat cross-run scalar equality as guaranteed by a fixed seed.

Open work: determine which shared residual sublayers/parameter updates initiate
the scale drift, verify optimizer/normalizer contracts directly, and use a
matched no-geometry continuation to isolate its historical cause. No active
training stop, hot edit, automatic control queue or validated recovery claim.


## Shared FFN rollback and natural recovery check

Additional detailed-layer audit localizes the main amplification to the shared
FFNs. Layer5 (zero-based) linear2 output RMS0.315→24.337, layer11 linear2
0.544→81.683; respective LayerNorm inputs to those FFNs remain approximately
unit RMS. Layer11 linear2's estimated top singular value grows5.57→11.64,
although its Frobenius norm only changes27.56→29.77. Scene-context estimated
top singular value1.60→4.78. These are16-iteration power estimates, not exact
SVDs. Parameter-state audit finds all4 inherited normalizer tensors exactly
unchanged,568 optimizer state counters at4500, no nonfinite state tensors or
negative second moments. Baseline b1821 versus launch b852 has no model.py or
normalizer implementation change.

Group rollbacks substitute only specified4371 parameters into disposable4500
copies; same four TRAIN inputs, no optimizer step, no model file write:

| Parameter group restored | Shared RMS | Context RMS | Scene gradient norm | Scene loss |
| --- | ---: | ---: | ---: | ---: |
| None | 423.14 | 3408.45 | 2432.54 | 0.3821 |
| Shared FFNs | 14.45 | 96.22 | 159.45 | 2.1741 |
| Shared attention | 355.62 | 2871.95 | 5979.51 | 0.6589 |
| Shared LayerNorms | 408.99 | 3293.36 | 1102.39 | 0.4136 |
| Scene context projection | 423.14 | 1017.42 | 6070.79 | 7.1351 |
| Entire shared Transformer | 3.78 | 11.45 | 47.30 | 0.5733 |
| Shared linear2 only | 55.73 | 305.96 | 101.59 | 3.7959 |
| Full source reference | 3.58 | 5.58 | 2.76 | 0.0341 |

This intervention supports the shared FFNs as a major source of residual-scale
amplification, not merely a correlation with scene gradients. It does not make
hybrid checkpoints accurate: changed groups interact with adapted downstream
weights, and partial restorations often raise the loss. Source weights are
preserved; no hybrid is proposed for execution or resumed training.

The active run naturally partially recovers by9000. Fixed32-GRAB DEV hand
EPE55.4514mm (initial52.6763), hand h24=101.362mm (initial102.509), object h24=
37.2623mm (initial15.8196), geometry-relative24.1056mm (initial22.5379).
Thus the7000 deterioration is not permanent monotone collapse, but there is
still no net object/relative-contact improvement. Four-TRAIN audit at9000 shows
shared RMS5.119, context16.740, scene-head gradient6.399 (FP32=6.388), compared
with hundreds/thousands during the unstable phase. Recovery of feature scale
and gradient conditioning co-occurs with physical error recovery.

Evidence under outputs/ptv3-shared-wam-v6/: shared-layer-diagnosis-20261021-r1,
shared-rollback-diagnosis-20261021-r1 (8 variants,17.31s, peak1.104GiB),
parameter-state-diagnosis-20261021-r1, recovery-9000-diagnosis-20261021-r1,
and instability-diagnosis-summary-20261021-r1/ (summary.json, curves.png).
Remaining historical attribution: new geometry loss, fresh optimizer and
continued adaptation are not isolated without matched continuation. The
observed failure path is shared FFN/residual scale growth → scene-context gain
→ ill-conditioned prediction/gradient path, aggravated by low precision.
A normalization insertion changes checkpoint behavior, and simple FP32 or
Adam reset has not established a recovery method. No production change made.

Related primary research on residual amplification:
[Understanding the Difficulty of Training Transformers](https://arxiv.org/abs/2004.08249).
This motivates the diagnostics; it does not prove our trigger or validate an
unrun repair. Local notes and source metadata are under
paper/transformer-training-difficulty/.

Final diagnostic verification: complete v6 suite31 tests PASS in8.00s,
new audit Python syntax and changed-document/paper-metadata checks PASS.
No production model/objective/trainer or active config changed in this diagnosis.


## PointWorld FFN source comparison

User asks to borrow FFN details without copying the full architecture. Clean
PointWorld0548482 and official source inspected. Core MLP/GELU/4x/pre-norm
mapping is the same; matched-weight CPU FP64 forward and input/parameter
backward smoke gives exactly zero differences. LayerScale is NOT enabled in
the release builder. PointWorld has scheduled residual DropPath up to0.3 and
independently initialized hierarchical blocks; our added shared global768/12
stack has no DropPath and starts from cloned equal-valued Torch layers.
The actual pretrained4371 layers are already distinct. These differences are
candidate comparisons, not proven causes or an excuse to reset learned weights.

Our geometry PTv3 already uses the vendor code (configured DropPath0.1), and
remains near its original feature scale in the measured examples. The diagnosis
concerns the additional shared stack. Missing final output normalization alone
was overemphasized previously: PointWorld also has pre-norm FFN residuals with
no required post-residual norm. It is a candidate stabilization, not a proven
root cause. Details/source metadata: paper/pointworld/ffn-comparison.md and
schema.json. Audit: outputs/ptv3-shared-wam-v6/pointworld-ffn-audit-20261021-r1/.
No active model, checkpoint or training configuration changed.

## Completed10000 and follow-up precision arm

Original BF16/TF32 run completed10000 normally, elapsed12907.30s, final
loss0.93622; final568-state/world3/RNG/normalizer checkpoint audit PASS.
GRAB fixed DEV final hand56.4268mm, hand-h24=105.1461mm,
object-h24=30.3106mm, geometry-relative19.4088mm. Initial corresponding
hand52.6763mm/object-h24=15.8196mm/geometry-relative22.5379mm: better geometric
relative error did not establish better tau/object prediction. Later natural
recovery does not remove the earlier FFN amplitude instability. GPUs4/5/6 have
released. User now requests a new strict-FP32 healthy4371 training arm; see
[P-20261021-wam-v6-strict-fp32](P-20261021-wam-v6-strict-fp32.md).
