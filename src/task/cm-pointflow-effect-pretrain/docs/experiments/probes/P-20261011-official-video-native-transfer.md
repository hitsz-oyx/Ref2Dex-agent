---
schema: ref2dex.probe.v2
probe_id: P-20261011-official-video-native-transfer
experiment_id: P-20261011-official-video-native-transfer
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 590fd98
claim_id: C1
hypothesis_family: official-video-data-only-native-transfer
probe_index_in_family: 1
seed_pool: probe
seeds: [228, 231]
decision_changed_if_positive: retain official H5 data-only pretraining as a native shared-scene initialization route
decision_changed_if_negative: close this explicitly native-compatible official-data initialization recipe while retaining the verified H5 adapter and other data contracts
status: UNPROMISING
run_id: official-video-native-transfer-20261011-r1
---

# Official video data-only native-transfer Probe

Class: Decision. Earlier official H5 runs established a valid random-initialized
short-horizon training path but did not test whether its learned scene
representation can initialize the existing native action-conditioned model.
This Probe closes that missing question without loading any official model
weights.

## Decision note

**Question:** does a shared scene projection/PTv3 backbone trained from the
official H5 data improve a matched native action-conditioned run over random
initialization?

**Evidence:** the H5 pack has finite metric 3-D points and ten future flow
steps, but its color/state channels do not have the native scene feature
semantics. A direct weight copy would therefore be invalid. The new pack maps
only context normals into native channel 0:3 and zeros unavailable causal
kinematics/identity channels; native train-only normalization is reused. The
native trainer copies only shape-compatible `scene_proj`, PTv3 core and scene
skip tensors; action, hand, object, output heads and native normalization stay
fresh.

**Cheapest test:** 500-update one-GPU matched `action` arms on the frozen
OakInk2/GRAB/ARCTIC data and validation panel, using seed 231 and the same
draw/config for random and video-data initialization. No official checkpoint,
ZED SDK, or additional corpus is used.

## Fixed protocol

- Official source: the pinned DROID/PointWorld H5 and its 8/2 clip split.
- Data pretraining: 500 random-initialized updates on the native-compatible
  H5 pack, using native `scene_mean/std`; `official_checkpoint_loaded=false`.
- Native transfer: copy only `scene_proj.*`, `backbone.core.*` (from the raw
  video PTv3 backbone), `skip_gamma` and `skip_beta`; reset optimizer/schedule.
- Native controls: same temporal trainer, config, seed, data manifest, stats,
  loss statistics, and 500 updates for random and video-initialized arms.
- Device: one idle GPU, with utilization, peak allocation and elapsed time
  recorded; output cap 3 GiB per arm. Native control/transfer seed is 228;
  official data pretraining seed is 231.

## Decision rule

The data-only route is `PROMISING` only if the video-initialized arm improves
the fixed moving/static validation macro by at least 10% at the final step and
does not worsen either component by more than 20%. Otherwise classify this
bounded single-seed transfer as `UNPROMISING` or `UNCLEAR` according to metric
availability. It cannot establish a final Cm policy benefit.

## Results

The native-compatible H5 pretraining completed 500 updates in 69.6 seconds on
GPU1 with 1,002 MiB peak allocation. It used the pinned H5 SHA256
`99bfb31a46c331fa49b87ca33f46fc17a5795c8f1785562e043122d0e27caa2b`, native
scene statistics, and `official_checkpoint_loaded=false`/`zed_used=false`.

The matched native arms used the same data manifest, draw hash, config, seed
228 and 500 updates. Each took about 4.4 minutes with 4,835 MiB peak
allocation. The fixed balanced validation panel was:

| arm | moving H24 point EPE | static H24 point EPE | moving/static macro |
|---|---:|---:|---:|
| random native | 101.296 mm | 5.977 mm | 0.053637 m |
| official H5 data-only scene init | 101.580 mm | 6.020 mm | 0.053801 m |

The data-only initialization is 0.3% worse on the fixed macro score, with no
10% improvement and no component-level gain. The Probe is therefore
`UNPROMISING` for this native-compatible feature mapping and 500-step transfer
recipe. The shared initializer itself passed the tensor-contract smoke: 454
shape-compatible tensors (about 50.4M parameters) were imported, while native
normalization and action/hand/object/output heads stayed fresh.

## Limitations / future evidence

The official H5 pack contains one matched DROID episode, and this transfer is
one seed with a short native budget. The negative result closes this feature
mapping and initialization recipe, not every possible official-video data
augmentation or multi-episode pretraining design. It also does not evaluate
downstream Cm-on/Cm-off policy benefit.
