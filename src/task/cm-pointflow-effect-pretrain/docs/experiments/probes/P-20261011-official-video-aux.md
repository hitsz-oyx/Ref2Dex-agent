---
schema: ref2dex.probe.v2
probe_id: P-20261011-official-video-aux
experiment_id: P-20261011-official-video-aux
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: c89d33e
claim_id: C1
hypothesis_family: official-video-scene-flow-auxiliary
probe_index_in_family: 1
seed_pool: probe
seeds: [230]
decision_changed_if_positive: prepare a matched shared-backbone transfer into the native Cm downstream task
decision_changed_if_negative: defer official H5 pretraining and retain the route only as a data-contract result
status: UNPROMISING
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

The reproducible run used the hash-locked 20-sample NPZ pack (8 temporal fit
clips and 2 held clips, two cameras), 512 all-step-valid tracks per sample,
seed 230, 500 updates, and GPU1. It completed in 67.4 s with peak allocation
1,003 MiB; the pack records H5 SHA256
`99bfb31a46c331fa49b87ca33f46fc17a5795c8f1785562e043122d0e27caa2b`. No
official checkpoint or ZED dependency was used.

| held metric | trained auxiliary | static persistence |
|---|---:|---:|
| H1 EPE | 0.979 mm | 0.722 mm |
| H5 EPE | 1.760 mm | 1.461 mm |
| H10 EPE | 2.263 mm | 1.878 mm |

The learned arm loses to the zero-flow persistence baseline at every reported
horizon and both held clips, so this exact color/normal-only recipe is
`UNPROMISING`. The training/data path itself completed normally. The negative
result is limited to a one-context, no-robot-state representation; it does not
reject the official H5 route or a state/action-conditioned auxiliary.

## Limitations / future evidence

This Probe establishes only official-H5 auxiliary learnability for a
color/normal-only input. It does not show native Cm utility, policy benefit,
cross-episode generalization, or that the feature projection is adequate. The
H5 robot state/action fields were intentionally not consumed here; a separate
state-conditioned card is required before closing the official-data route. A
positive result on that route would still require a matched native downstream
transfer with random and data-pretrained shared backbones.
