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
seed_pool: probe
seeds: [228]
decision_changed_if_positive: expand matched native transfer from official video-pretrained PointWorld backbone
decision_changed_if_negative: inspect pretrained feature compatibility before larger video investment
status: UNCLEAR
run_id: open-video-backbone-transfer-20261010-r1
---

# Does the released video-pretrained backbone help our native effect model?

Class: Decision. User requests open-source reuse before writing new pipelines;
the immediate question is video benefit to this model, not reproduction of a
different video task. Mission remains action-conditioned Cm and eventual matched
Cm-on/off benefit to self-trained robot RL.

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

Primary: equally weighted source macro of h24 EPE on
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

## Results

Official acquisition completed within the15min bound. Full SHA256 and byte
count match the pinned asset. Strict CPU preflight passed:448/448 tensors,
50,417,280 entries, zero missing/extra keys, shape/dtype mismatches or nonfinite
weights. Inspection took5.928s; no neural inference/training was run on CPU.
Compatibility reports are preserved in the preflight output above.

The minimal native-format initializer conversion uses existing model/data code;
no new training loop. Two targeted integrity tests passed (8.81s), checking that
only the spatial backbone changes and rejecting partial/nonfinite/mismatched
imports before mutation. Initializer export passed. Both artifacts preserve exactly the same nonbackbone
state SHA25639b8fba696a17d4617bb35ded55fbf4e4034386a8bb903273d3dd6914683643f;
only the complete spatial backbone changes. Outputs are202,183,941 and
202,159,518 bytes, with no native training/optimizer history. Three-update GPU2
engineering checks passed for both arms (6.476s/6.772s), including identical
constructor/draw/source/stat/config hashes. Stable updates about0.4s, peak
reserved4,686MiB; benchmark skips checkpoints/evaluation. Full engineering
metadata lives in `engineering_summary.json` and `initialization-r1/manifest.json`.

Paired2000-update training started on free GPUs2(random) and3(video), using
unchanged existing launcher/trainer at Git199c7db. Per-arm2700s deadline,6GiB
group artifact bound. Arm roots `random-r1/` and `video-r1/`; authoritative live
status/PIDs/deadlines in each `group_status.json`, model identities/progress in
`train-action/`. Startup estimate15–25min including validation. No benefit
result or change of Mission claim yet.

## Limitations / future evidence

DROID is real robot demonstration video, not human egocentric video. This Probe
asks about that released video-pretrained spatial backbone; it cannot attribute
benefit to video alone versus other details of the author's pretraining recipe,
or establish EgoDex/EPIC/EgoTouch benefit. Native feature/action/effect interfaces
remain different from the official RGB-D/gripper model. Native attention
patch_size128 and time-aware pooling are retained; source patch_size256 and
ordinary spatial pooling differ despite exact tensor compatibility. Single seed and short
matched training cannot establish a formal advantage; multi-seed matched budgets
and eventual robot Cm-on/off evidence are deferred until a positive signal.
