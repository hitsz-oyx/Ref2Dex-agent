---
schema: ref2dex.probe.v2
probe_id: P-20261010-video-point-dynamics
experiment_id: P-20261010-video-point-dynamics
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: pending
claim_id: C1
hypothesis_family: HF-video-point-dynamics
probe_index_in_family: 1
seed_pool: probe
seeds: [226]
decision_changed_if_positive: expand qualified video tracks and test encoder transfer to native hand-conditioned dynamics
decision_changed_if_negative: diagnose track quality or missing interaction inputs before scaling video training
status: UNCLEAR
run_id: video-point-dynamics-20261010-r1
---

# Video point prediction without dense future hand labels

Class: Decision. The global goal remains action-conditioned Cm for self-trained
robot manipulation; video history prediction is an intermediate representation
pretraining test, not a replacement for action conditioning or policy validation.

Question: Does the existing PTv3 encoder plus a point-displacement head learn
held-source-video motion information from weak RGB/depth tracks beyond static
and constant-velocity baselines? A positive result justifies expanding video
labels and testing transfer into native geometry training; a negative result
blocks blind scaling and prioritizes label/interaction-input diagnosis.

## Design / decision note (pre-run)

Dense Contact hands have zero accepted local H4+K24 windows. Choose a separate
video-only loader/head that never reads Contact, FoundationPose, TRELLIS, future
hands or tactile targets. Retain the existing native model/data/checkpoints.
Reuse PTv3-small topology and the native 18-component scene feature meanings;
missing surface normals are explicitly zero. Predict per-point future flow
rather than pretend the clip has rigid-object GT. Train from random
initialization; a transfer comparison is a later Decision gate.

Data: already-local ObjectForesight shards, require clock metadata and official
TRAIN object membership. All P03_03 clips are eligible for TRAIN selection;
all P03_13 clips are held for development only. P01_03 missing-clock clips are
excluded. Choose up to4 nonoverlapping clips per video in stable sorted order;
require55..600 source frames, no ambiguous camera convention (>=5x static
reprojection separation). Source videos and selected action time intervals may
not overlap across partitions. Development is a held subset of original TRAIN
material, not an official test score. Contact-test frames are never training
inputs. If qualification leaves fewer than2 clips per partition, do not train.

Tracks: fixed source stride2, record actual29.97002997Hz and original frame IDs.
RGB LK requires forward/backward error <=1 pixel and object mask membership;
failed IDs never revive. Background is frame-zero metric depth, with per-frame
depth/mask occlusion checks; retain measured depth jitter, not forced zero flow.
Initial track candidates are selected from frame-zero observations only. For
each H4+K24 window, input points are selected using all four HISTORY validity
frames only, with up to128 object and128 background points. Future label masks
may affect loss/metric support but never input point selection or features.

Features: history velocity/acceleration use actual timestamps, history
displacement and static indicator use history only; anchor frame is a fixed
current-point-derived center, independent of future labels. Normalization uses
TRAIN inputs only. Future flow loss uses a fixed0.01m Huber scale; object and
background losses each receive half the mass when both are present. Each
window/group is averaged before pooling windows, so abundant background or
longer valid tracks cannot swamp object dynamics. All invalid target entries
are excluded before arithmetic. Report supported counts and physical-metre EPE.

Controls: zero-flow/static and constant velocity from the last two history
frames at actual elapsed time. Forward receives xyz/features/input validity
only; future labels/masks, point kind and split identifiers are absent from its
interface. Test future-label changes leave model inputs and outputs unchanged,
and invalid labels do not affect loss. Verify CUDA backward/checkpoint roundtrip
before a non-smoke run.

Training: one currently free GPU, seed226, batch2, max500 updates, AdamW lr1e-4,
weight decay.01, grad clip1, BF16 where supported, fixed final checkpoint only.
Development evaluation every100 updates is diagnostic; no threshold tuning or
best-step selection. Decision criterion: final development object-track h24 EPE
improves >=10% over the better of static/CV, while background h24 EPE is <=1.2x
static. Report all-horizon EPE as well. No scientific conclusion from training
loss or a fixed-batch overfit; results are Probe labels only.

Budget: CPU preparation<=20min, one GPU<=15min, new artifacts<=5GiB total,
raw extraction<=2GiB. Stop on nonfinite data/loss, source/hash drift, collision,
GPU conflict, resource cap or deadline; preserve failed run/log. Inspect GPU
utilization/memory after startup and estimate ETA. No system modifications,
remote training, full download, existing checkpoint overwrite or new branch.

Outputs: `outputs/cm-pointflow-effect-pretrain/video-point-dynamics-20261010-r1/`.

## Results

Pending implementation/qualification at a fixed execution commit.

## Limitations / future evidence

Two source videos and overlapping temporal anchors cannot establish large-scale
or independent-episode generalization. LK consistency is not ground-truth
physical correspondence; static depth jitter remains label noise. Future
visibility masks condition scored support and must be reported. History-only
predictability does not demonstrate human-action causality or robot usefulness.
Positive evidence still needs native encoder-transfer and matched conditioning
controls, then the global Mission's self-trained robot policy comparison.
