---
schema: ref2dex.probe.v2
probe_id: P-20261010-droid100-native-adapter-contract
experiment_id: P-20261010-droid100-native-adapter-contract
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 62aa70b
claim_id: C1
hypothesis_family: official-video-pointflow-adapter
probe_index_in_family: 2
seed_pool: probe
seeds: []
decision_changed_if_positive: run a bounded random-initialization Probe on the official video/annotation pair
decision_changed_if_negative: close the direct native adapter route and choose an explicit temporal or RGB adapter
requested_intervention: official_video_data_training
executed_intervention: official_droid100_native_contract_audit
status: UNPROMISING
run_id: droid100-native-adapter-contract-20261010-r1
---

# Does the matched official H5 clip satisfy the native point-flow input contract?

Class: Decision. The preceding audits established that DROID-100 contains
official RGB/action records and that its first episode overlaps a pinned
PointWorld-DROID flow shard. This read-only spike checks the actual temporal,
point-identity, feature and action contracts before any native training. It
does not modify the trainer, download raw MP4/SVO, use a released checkpoint,
or start a GPU run.

## Frozen input and audit

Input is the restored episode
`RAIL+80edfcb1+2023-04-17-14h-48m-05s_flows.h5` from the pinned flow shard
`000409`, package SHA256
`38fc5fc032f422a34d62a5b7a6d554189f68fafdab0f568660651467de2b23e9`. The
current native video learner requires four history points and 24 future points
(`H=4,K=24`), observed `xyz/features/point_valid`, and an 18-D feature vector.
The one-time audit writes only the ignored JSON artifact
`tmp/droid100-adapter-audit-20261010.json`.

## Result

The matched episode has ten clips and two external cameras per clip. Every
camera has finite `scene_flows` with 11 time points and 23,337--27,162 points,
plus initial RGB/depth, colors, normals, visibility and camera calibration.
Thus zero clips provide the native 28 total time points; each has at most seven
future points after a four-point history.

The remaining native fields cannot be asserted from this release:

* `scene_colors` and `scene_normals` can fill only part of the 18-D scene
  feature vector. Velocity, acceleration and history displacement can be
  derived within one clip, but the required object/background `point_kind` is
  not published as a verified point label.
* Clip-level `gripper_pose` is 7-D and `joint_positions` is 7-D. They do not
  provide the native per-step 2-hand, 11-keypoint, 9-D action tensor.
* Overlapping clips have different point counts and the release provides no
  verified cross-clip point identity. Stitching them to manufacture 28 frames
  would add an unvalidated assumption.

The decision is `UNPROMISING` for a **direct native adapter** and
`INCOMPATIBLE_DIRECT_NATIVE_CONTRACT` in the audit JSON. This is a data-contract
finding, not evidence against the official video or PointWorld labels. No
training loss, policy metric or checkpoint result was produced.

## Next decision

Do not start a native training Probe from these H5 clips. A separate [raw temporal audit](P-20261010-droid100-raw-video-temporal-audit.md)
shows that the
source MP4 itself has 166 frames at 60 FPS, so one possible route is to run the
official processing pipeline with a longer clip. That route still needs
verified point identities and native hand/action labels. The other options are
an explicitly shorter PointWorld-style horizon or a separate RGB/robot-state
front-end. Each route needs its own matched random-initialization control and
experiment card. The current official-data route remains available, but the
released H5 shard cannot be silently presented as native `H=4,K=24` training
data.
