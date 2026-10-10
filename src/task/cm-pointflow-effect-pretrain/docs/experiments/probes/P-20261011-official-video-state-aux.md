---
schema: ref2dex.probe.v2
probe_id: P-20261011-official-video-state-aux
experiment_id: P-20261011-official-video-state-aux
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: c25a3b7
claim_id: C1
hypothesis_family: official-video-state-conditioned-scene-flow
probe_index_in_family: 2
seed_pool: probe
seeds: [231]
decision_changed_if_positive: retain official H5 video plus robot-state auxiliary for matched native backbone transfer
decision_changed_if_negative: close this H5 state-conditioned recipe while keeping RBO and other official RGB routes separate
status: PLANNED
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

Pending execution.

## Limitations / future evidence

This is still one DROID episode and a short auxiliary task. A positive result
only justifies a matched native transfer Probe; it does not establish Cm policy
utility or cross-episode generalization. A negative result closes this
state-conditioned H5 recipe, not RBO measured wrench data or official RGB/state
front-end research in general.
