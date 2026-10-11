---
schema: ref2dex.probe.v2
probe_id: P-20261010-open-video-backbone-transfer
experiment_id: P-20261010-open-video-backbone-transfer
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 199c7dbd9ea14b761acabac89fd852a3fee442ee
claim_id: C1
hypothesis_family: HF-open-video-backbone-transfer
probe_index_in_family: 1
requested_intervention: official_video_data_training
executed_intervention: official_pretrained_backbone_initialization
seed_pool: probe
seeds: [228]
decision_changed_if_positive: expand matched native transfer from official video-pretrained PointWorld backbone
decision_changed_if_negative: inspect pretrained feature compatibility before larger video investment
validity: INVALID_IMPLEMENTATION
scientific_conclusion: INCONCLUSIVE
status: UNCLEAR
run_id: open-video-backbone-transfer-20261010-r1
---

# Historical checkpoint-transfer run (not the requested official-video-data test)

Class: Decision. User requests open-source reuse before writing new pipelines;
the immediate question is video benefit to this model, not reproduction of a
different video task. Mission remains action-conditioned Cm and eventual matched
Cm-on/off benefit to self-trained robot RL.

Validity correction: **INVALID_IMPLEMENTATION for the intended official-video-data
question**. The user intended to train with released video data, but this run
only initialized the native model from an official pretrained checkpoint. The
training data remained the native OakInk2/GRAB/ARCTIC manifest; no official DROID
video data entered training. The checkpoint-transfer measurements are retained
as an engineering record, but they are **INCONCLUSIVE** for video-data benefit.
Decision: do not use this run to stop the video-data route. Preserve its outputs,
reclassify its transfer gate as recipe-level evidence only, and perform a
separate bounded audit of the released video data before any acquisition or
training.

## Actual protocol executed (checkpoint-transfer interpretation)

The following protocol records the intervention that was actually executed after
the open-source request was misinterpreted. It is retained for reproducibility;
it is not the protocol for the intended official-video-data experiment.

## Decision note / frozen protocol before training

Official PointWorld small-DROID checkpoint is published and shares the native
PTv3-small blueprint. Use it to initialize only the entire spatial backbone,
including its parameter/buffer state. Do not copy RGB/DINO, robot projections,
normalizers, optimizer, temporal/action embeddings or effect head. The native
time-aware pooling wrapper remains. All backbone tensors must correspond by
name and exact shape; incomplete/ambiguous mapping aborts. If compatible,
prepare two native-format weights-only initializer artifacts and use the existing
`--init-weights` interface for both arms. This avoids changing the trainer or
launcher and gives both arms the same pre-training evaluation/RNG path. No new
model, video tracker, dataset converter or training loop.

Checkpoint repo `nvidia/PointWorld_models`, revision
`b9e2e19a4f2bd65922e1f6d70aa953fe70aa9dba`, file
`small-droid/model-best.pt`, bytes1,826,853,514, SHA256
`ccb9ed93dff5eea976010c57dd0cb5634db61c68b732c4437cbf54c8da9de8fe`.
Acquisition/CPU compatibility is engineering preflight under
`outputs/cm-pointflow-effect-pretrain/open-pointworld-small-preflight-20261010-r1/`;
it does not establish learning benefit. Its3GiB/15min bounds remain separate.

Two independently trained action-conditioned native models: random backbone
vs released video backbone, seed228. Model construction seed and all nonbackbone
parameters/buffers are identical before training. Both fully fine-tune their
backbones. Fresh AdamW states, same LR/schedule, seed/draw, TRAIN data, statistics,
target/loss/validation panel and fixed final step. Preserve exact sample draw,
input/stat/config/source hashes and initial nonbackbone parameter hashes.

Use the retained shared-stat OakInk2/GRAB/ARCTIC manifest, with4,378,478 TRAIN
anchors, source weights5/9,2/9,2/9, native H4/K24 at30Hz, fixed0.01m voxel origin
[-2,-2,-2], temporal pooling, per-horizon action summary, motion_floor0.1, and
train-only physical loss statistics. No weak EPIC/EgoTouch rows enter training.

Recipe:2000 updates, per-rank microbatch16, accumulation2, one GPU per arm,
effective batch32, LR1e-4, weight_decay0.01, warmup100, clip_grad1, BF16 AMP.
Validation96 windows, balanced seed212 and natural seed213, validation and
checkpoint intervals500, final checkpoint only for the gate. Intermediate
validation is diagnostic, no best-step selection. The historical50000-update
native result is background, not a matched random-init control.

Primary on the fixed balanced96-window panel: equally weighted source macro of h24 EPE on
`model/moving_objects/cat-1` and separately `model/static_objects/cat-1`, with per-source
and pooled values retained. Source panels/stratum label counts must match both
arms. Positive gate: moving EPE at least10% below the new matched random arm,
static EPE no greater than1.2x random. Report all-horizon/source metrics and
actual-vs-shuffled action final intervention; actual hands are still retrospective
human oracles, not evidence of controllable robot interventions. Training loss,
compatibility, or author benchmark scores cannot pass this gate.

Budget: at most2 simultaneously free GPUs,45min per arm, training artifacts6GiB
total (separate from acquired official checkpoint), smoke at most3 updates per
arm on one free GPU. Inspect GPU load/memory and ETA after startup. Stop on
source/hash drift, nonfinite loss, output collision, resource/deadline breach,
unknown process on assigned GPU, or incomplete mapping. Do not preempt unknown
jobs, overwrite retained checkpoints, push or create a new branch. No new
external authorization needed within current Campaign/user scope.

Positive: consider a longer matched run before native transfer or RL claims.
Negative: do not add video corpus/epochs blindly; inspect missing appearance and
different input feature distributions. A negative short Probe only rejects this
backbone-only transfer recipe, not all video or human-video pretraining.

## Inputs and outputs

Data `outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data/`.
Forward/loss stats are sibling `norm_stats.json` and `loss_stats.json`.
New config/implementation hashes and actual execution Git commit will be frozen
after compatibility preflight, before non-smoke training.

Output `outputs/cm-pointflow-effect-pretrain/open-video-backbone-transfer-20261010-r1/`.

## Results (checkpoint transfer only; not the intended video-data experiment)

Official acquisition and compatibility preflight completed within the frozen
bounds. Both initializer integrity tests and both three-update GPU checks passed.
The two matched training arms then completed the planned 2000 updates with exit
code 0: random-backbone 15.53 minutes and released-video-backbone 15.72 minutes.
The final result, validation arrays, logs and checkpoints are preserved under
`outputs/cm-pointflow-effect-pretrain/open-video-backbone-transfer-20261010-r1/`.
The full all-horizon/source extraction and gate calculation is in
`result_analysis_r1.json`.

The final identity audit passed. Both arms used the same Git commit
`199c7dbd9ea14b761acabac89fd852a3fee442ee`, dataset/config/statistics/loss/source
hashes, seed 228, effective batch 32, global draw
`92babbdf52a1178aa2c4666d8229b36565683980e4e6b9ec631150de3005a139`, validation
panel and trainer. The only intended initialization difference was the complete
spatial backbone; all metrics were finite, and balanced/natural/shuffle panels
have identical metric keys. No assigned training process remains on GPU2/3.

On the fixed balanced source-macro panel, the predeclared positive gate fails both criteria:

| Primary final h24 point EPE | Random backbone | Video backbone | Change | Gate |
| --- | ---: | ---: | ---: | --- |
| Moving objects, source-macro `cat-1` | 0.080660 | 0.082070 | +1.75% | video must be ≤0.072594 (fail) |
| Static objects, source-macro `cat-1` | 0.022597 | 0.029279 | +29.57% | video must be ≤0.027117 (fail) |

The per-source h24 moving EPE changes are +8.65% (OakInk2), +0.19%
(GRAB), and +3.46% (ARCTIC). Static h24 changes are +82.61%, +54.09%, and
−6.82%, respectively. The JSON artifact retains every h1/h4/h8/h12/h24,
source and stratum value; these source differences do not rescue the fixed
source-macro gate.

The actual-versus-shuffled-action intervention is retained as a diagnostic. At
h24 moving-object EPE, shuffling actions raises error from 0.080660 to 0.115215
(+42.84%) for the random arm and from 0.082070 to 0.114830 (+39.92%) for the
video arm. This confirms action sensitivity in both trained models, but gives no
video-transfer advantage. Static-object shuffled EPE falls at h24 in both arms
(−26.13% and −28.65%), so it is not evidence of a useful static action effect.

The checkpoint-transfer recipe itself did not pass its predeclared gate, but that
result cannot answer whether the released video data help the native model. The
run is therefore **INVALID_IMPLEMENTATION / INCONCLUSIVE** relative to the
intended data intervention. It must not be used to reject official DROID data,
video pretraining, or human-video pretraining. No official DROID video file was
used by either training arm; the only official asset was the 1.83 GB checkpoint.

The post-run implementation audit confirms that the released backbone was the
intended intervention. Both final checkpoints changed all 448 backbone tensors
relative to their initializer (relative backbone L2 change 0.0749 for random
and 0.0530 for video); the 26 trainable nonbackbone tensors also changed in
both arms. The direct transfer nevertheless crosses an input-contract gap:
the released checkpoint records 31-dimensional scene features, 0.015m grid
size and patch size 256, while the native arm feeds 18-dimensional projected
features on a 0.01m grid with patch size 128 and temporal pooling. Exact tensor
shape compatibility therefore proves only that the weights can be loaded; it
does not make the two backbone input distributions equivalent. This is a
limitation of the attempted checkpoint-transfer recipe, not evidence about
training on the official video data.

## Limitations / future evidence

DROID is real robot demonstration video, not human egocentric video. This run did
not train on DROID and therefore provides no evidence for or against its data.
Native feature/action/effect interfaces remain different from the official
RGB-D/gripper model. Native attention patch_size128 and time-aware pooling are
retained; source patch_size256 and ordinary spatial pooling differ despite exact
tensor compatibility. The original checkpoint-transfer outputs remain useful
for tracing the failed intervention, while a data-use experiment needs its own
card, data manifest, adapter, control and gate.
