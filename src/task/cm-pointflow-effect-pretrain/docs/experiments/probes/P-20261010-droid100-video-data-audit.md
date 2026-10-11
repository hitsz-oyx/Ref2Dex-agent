---
schema: ref2dex.probe.v2
probe_id: P-20261010-droid100-video-data-audit
experiment_id: P-20261010-droid100-video-data-audit
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 7a66aac
claim_id: C1
hypothesis_family: official-raw-video-data
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: run a bounded adapter audit on an official video/annotation pair
decision_changed_if_negative: stop official DROID acquisition and retain the native data route
requested_intervention: official_video_data_training
executed_intervention: official_droid100_video_data_audit
status: PROMISING
run_id: droid100-video-data-audit-20261010-r1
---

# Can the official DROID video subset provide actual frame/action samples?

Class: Decision. The earlier `PointWorld-DROID` acquisition is an official
PointWorld-derived annotation package, not the original DROID video corpus. This
bounded source audit checks the official DROID-100 RLDS route before any model
training. It must expose actual JPEG frame fields and synchronized robot/action
records; a successful audit permits a separate small data-adapter Probe.

## Decision note / frozen scope

The official DROID documentation advertises `droid_100` as a roughly 2 GB,
100-episode RLDS debugging subset, while raw stereo MP4 data is multi-terabyte.
Use one public TFRecord shard first (two episodes, about 27 MB) plus the official
`dataset_info.json` and `features.json`. This is enough to distinguish a real
video-bearing record from metadata or derived PointWorld labels. Do not download
the full raw bucket, the full DROID RLDS corpus, or any PointWorld checkpoint in
this run. Stop on source/hash drift, output collision, schema mismatch, or any
request for remote NAS/full-corpus storage.

## Inputs and outputs

Source bucket: `gs://gresearch/robotics/droid_100/1.0.0`; expected objects are
`dataset_info.json`, `features.json`, and
`r2d2_faceblur-train.tfrecord-00000-of-00031`. The features contract is expected
to contain `wrist_image_left`, `exterior_image_1_left`, and
`exterior_image_2_left` as 180x320 JPEG frames, alongside robot state, action,
and language fields. Output root:
`outputs/cm-pointflow-effect-pretrain/droid100-video-data-audit-20261010-r1/`.

The audit records object sizes/hashes, TFRecord integrity, episode/step counts,
JPEG decode dimensions, and the minimal adapter contract. It does not train or
claim that video improves the native model.

## Result

The GCS inventory contains 33 objects: 31 TFRecord shards, `dataset_info.json`,
and `features.json`. The TFRecord objects total 2,192,595,669 bytes and the
metadata reports 100 episodes across shard lengths summing to 100. The first
shard is 27,488,816 bytes and its GCS MD5 is
`IQoCwa04u58UpSOzKsS/6g==`; the local copy has the same MD5 and SHA256
`37d04d3586fbde386d14122320d8f7d866b37007ff7a8d96625e72a3994dda70`.

The first shard contains two valid `SequenceExample` records with 166 and 238
steps. Every sampled first/middle/last frame in all three camera fields
(`wrist_image_left`, `exterior_image_1_left`, and `exterior_image_2_left`)
decoded as RGB JPEG at 320x180. The same records contain 7-D actions, 7-D joint
positions, boundary flags, language instructions, and source recording paths.
The full parsed result is in the ignored run artifact
`audit_result.json`; the object inventory is `source/inventory.json`.

This is a valid, bounded official video-data source (`PROMISING` for data
readiness). It is an RLDS TFRecord release with downsampled JPEG frames, not the
multi-terabyte raw MP4/SVO bucket. It therefore proves that official video
samples can be acquired and decoded, but it does not yet prove usefulness for
the native point-flow model.

## Next decision

The native model currently consumes 3-D point trajectories, 18-D scene features,
and action-conditioned targets; the DROID-100 RLDS record supplies RGB frames,
robot states and actions but no 3-D point-flow labels. One DROID-100 episode
has now been matched to the released PointWorld-DROID flow annotation in
[the pairing Probe](P-20261010-droid100-pointworld-match.md): the exact episode
is present in flow shard `000409`, with ten clips, two matching camera serials,
and finite 3-D scene-flow fields. This verifies a concrete official video plus
annotation source without using any released checkpoint. An RGB-only adapter
remains a separate route and must not be conflated with native point-flow
training. The follow-up [native-contract Probe](P-20261010-droid100-native-adapter-contract.md)
found that these clips cannot be passed directly to the current `H=4,K=24`
learner, so no training route is opened by this audit.
