---
schema: ref2dex.probe.v2
probe_id: P-20261011-official-video-aux
experiment_id: P-20261011-official-video-aux
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: pending
claim_id: C1
hypothesis_family: official-video-scene-flow-auxiliary
probe_index_in_family: 1
seed_pool: probe
seeds: [230]
decision_changed_if_positive: prepare a matched shared-backbone transfer into the native Cm downstream task
decision_changed_if_negative: defer official H5 pretraining and retain the route only as a data-contract result
status: PLANNED
run_id: official-video-aux-20261011-r1
---

# Official DROID / PointWorld H5 auxiliary Probe

Class: Decision. This is the requested data-only intervention: train from
random initialization on the already restored official PointWorld-DROID H5
scene-flow data. It does not load the published PointWorld model weights and
does not pretend the 11-step H5 clip is the native H=4/K=24 contract.

## Decision note

**Question:** does a short-horizon model learn useful scene-flow structure from
official video-derived H5 labels when initialized randomly?

**Evidence:** the pinned episode has ten clips, two external cameras, one
context scene and ten future flow steps, with finite visibility/depth masks,
colors, normals, and robot state. The prior direct native adapter was
`UNPROMISING` only because it demanded H=4/K=24, point-kind labels, and native
hand-action tensors.

**Cheapest test:** use the fixed 10-clip H5 episode, keep 8 temporal clips for
fit and 2 for held evaluation, sample 512 tracks valid for all 11 steps from
each camera, and train the existing PTv3-small video point head for 500 GPU
updates. Compare the learned H10 EPE with the same held target's zero-flow
persistence. A positive result only opens a later shared-backbone transfer
Probe.

## Fixed protocol

- Input H5 (materialized into a hash-locked NPZ pack before GPU execution):
  `outputs/cm-pointflow-effect-pretrain/droid100-pointworld-match-20261010-r1/restored/droid/flows-fs-optimized/RAIL+80edfcb1+2023-04-17-14h-48m-05s_flows.h5`.
- The H5 SHA256 is recorded in the run manifest before training; source files
  remain read-only. No official checkpoint, RGB/DINO weights, or ZED SDK is
  used.
- `prepare_official_video_aux.py` performs the only H5 read in a compatible
  HDF5 environment; `train_official_video_aux.py` consumes only the resulting
  pack and the existing random-initialized PTv3 model.
- Contract: context `scene_flows[0]`, targets
  `scene_flows[1:11] - scene_flows[0]`, explicit per-step visibility and
  depth-valid masks.
- Features: context color and normal channels plus documented zero
  placeholders to the current 18-D scene width; train-only mean/std.
- Selection: deterministic 512 all-step-valid tracks per camera and clip.
- Fit/held: first eight fixed clip keys for fit, last two for held; the held
  clips are from the same episode, so this is a learnability Probe rather than
  an independent-task validation.
- Compute: seed 230, 500 updates, matched static baseline, one idle GPU, at
  most 15 minutes and 3 GiB output. GPU utilization and peak memory are logged.

## Results

Pending execution.

## Limitations / future evidence

This Probe establishes only official-H5 auxiliary learnability. It does not
show native Cm utility, policy benefit, cross-episode generalization, or that
the color/normal-plus-zero feature projection is the best representation. A
positive result requires a new matched native downstream transfer with random
and data-pretrained shared backbones; a negative result does not reject the
official raw RGB/robot-state route or other calibrated data adapters.
