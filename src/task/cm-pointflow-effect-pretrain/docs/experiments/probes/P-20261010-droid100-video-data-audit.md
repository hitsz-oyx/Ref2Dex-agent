---
schema: ref2dex.probe.v2
probe_id: P-20261010-droid100-video-data-audit
experiment_id: P-20261010-droid100-video-data-audit
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 71ca378
claim_id: C1
hypothesis_family: official-raw-video-data
probe_index_in_family: 1
requested_intervention: official_video_data_training
executed_intervention: official_droid100_video_data_audit
status: UNCLEAR
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

## Current status

Metadata and one TFRecord shard have not yet been acquired in this card. The
previous PointWorld-DROID derived annotation audit remains preserved separately
and must not be counted as this raw-video audit.
