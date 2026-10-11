---
schema: ref2dex.probe.v2
probe_id: P-20261011-official-video-state-aux
experiment_id: P-20261011-official-video-state-aux
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 10fd8e0
claim_id: C1
hypothesis_family: official-video-state-conditioned-scene-flow
probe_index_in_family: 2
seed_pool: probe
seeds: [231]
decision_changed_if_positive: retain official H5 video plus robot-state auxiliary for matched native backbone transfer
decision_changed_if_negative: close this H5 state-conditioned recipe while keeping RBO and other official RGB routes separate
status: UNPROMISING
run_id: official-video-state-aux-20261011-r1
---

# Official H5 state-conditioned auxiliary Probe

Class: Decision. This follows the color/normal-only H5 Probe and consumes the
same official video-derived scene-flow labels, now conditioning the context
scene on the H5 `joint_positions[0]` and the first five components of
`gripper_pose[0]`. It remains a random-initialization data-only run.

## Decision note

**Question:** does adding the context robot state expose short-horizon motion
information that a one-context color/normal-only model cannot learn?

**Evidence:** the prior official-data Probe lost to persistence while ignoring
the H5 robot state. The H5 contract contains 7-D joint positions and 7-D
gripper pose at every one of the 11 scene-flow steps. Only context values are
used here; no future state enters features or normalization.

**Cheapest test:** reuse the pinned 10-clip H5 episode and the same 8/2
temporal split, 512 all-step-valid tracks, seed 231, 500 GPU updates, and
static held baseline. Change only the six remaining feature channels populated
by context joint/gripper state.

## Fixed protocol

- Source H5 and hash are inherited from
  `P-20261011-official-video-aux`; preparation writes a new immutable pack.
- Features are context RGB color, normal, 7-D joint position, and first 5-D
  gripper pose in the current 18-D width; train-only mean/std are used.
- Target remains `scene_flows[1:11] - scene_flows[0]` with per-step
  visibility/depth-valid masks.
- No official checkpoint, RGB/DINO weight, raw SVO, or ZED SDK is used.
- One idle GPU, 500 updates, maximum 15 minutes and 3 GiB output; utilization
  and peak memory are recorded.

## Results

The protocol-compliant run used seed 231, the immutable state-feature pack,
the same 8/2 temporal split and 20 samples, 512 tracks per sample, and 500
updates on GPU1. It completed in 70.5 s with peak allocation 1,003 MiB. No
official checkpoint or ZED dependency was used; the H5 SHA256 remained
`99bfb31a46c331fa49b87ca33f46fc17a5795c8f1785562e043122d0e27caa2b`.

| held metric | state-conditioned model | static persistence |
|---|---:|---:|
| H1 EPE | 1.021 mm | 0.722 mm |
| H5 EPE | 1.724 mm | 1.461 mm |
| H10 EPE | 2.183 mm | 1.878 mm |

The state-conditioned arm loses to persistence at every horizon and on both
held clips. The result is `UNPROMISING` for this one-context official H5
state-conditioned auxiliary recipe. The data adapter and random-initialized
training path remain valid; the negative result is not evidence that all
official video or robot-state pretraining is impossible.

## Limitations / future evidence

This is still one DROID episode and a short auxiliary task. The held clips are
temporal tails from the same episode, not independent tasks. This result closes
the current state-conditioned H5 recipe for further native transfer investment;
it does not establish Cm policy utility, cross-episode generalization, or a
negative result for RBO measured wrench data, CHORD simulator support, or a
different officially qualified RGB/state front end.
