---
schema: ref2dex.probe.v2
probe_id: P-20261010-droid100-pointworld-match
experiment_id: P-20261010-droid100-pointworld-match
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 7a66aac
claim_id: C1
hypothesis_family: official-video-pointflow-adapter
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: write and smoke-test a native point-flow adapter on the matched clip
decision_changed_if_negative: stop PointWorld pairing and keep official RGB as a separate route
requested_intervention: official_video_data_training
executed_intervention: official_droid100_pointworld_annotation_match
status: PROMISING
run_id: droid100-pointworld-match-20261010-r1
---

# Can an official DROID video episode be paired with PointWorld 3-D labels?

Class: Decision. The DROID-100 audit proved that official RGB frames and actions
are readable, but the native point-flow model requires 3-D trajectories and
scene features. The PointWorld flow manifest contains the same episode ID as
the first DROID-100 TFRecord record. This audit checks whether the released
processed H5 can be restored and paired with the video record, without loading
any PointWorld checkpoint or starting training.

## Decision note / frozen scope

Acquire only PointWorld-DROID flow shard `shard-000409`, whose manifest entry
contains `RAIL+80edfcb1+2023-04-17-14h-48m-05s_flows.h5`; its compressed part is
expected to be about 3.50 GB and its restored member bytes about 4.46 GB. Reuse
the already verified camera and confidence packages from the bounded audit when
possible, but do not overwrite that run. Stop on revision/checksum drift,
missing episode overlap, H5 schema mismatch, output collision, or total new
storage above 20 GB. No model weights, GPU training, full DROID corpus, or
remote NAS write is part of this Probe.

## Inputs and outputs

The video side is the verified DROID-100 record in
`outputs/cm-pointflow-effect-pretrain/droid100-video-data-audit-20261010-r1/`.
The processed side is `nvidia/PointWorld-DROID`, revision
`dd9aaeec94bb14e27ab6b16b6e4aa0dbcf3ef56f`, flow shard `000409`, restored with
the official helper. The new output root is
`outputs/cm-pointflow-effect-pretrain/droid100-pointworld-match-20261010-r1/`.

The audit records exact package hashes/sizes, matching episode and clip keys,
H5 fields and shapes, finite 3-D trajectories, action/robot-flow availability,
and RGB dimensional compatibility. A successful match only permits a later
small data-adapter training Probe; it is not evidence that video improves the
native model.

## Result

The pinned flow manifest identifies the DROID-100 first record
`RAIL+80edfcb1+2023-04-17-14h-48m-05s` in `shard-000409`. The downloaded part is
exactly 3,499,128,679 bytes with SHA256
`38fc5fc032f422a34d62a5b7a6d554189f68fafdab0f568660651467de2b23e9`. The
official recovery helper restored 68 flow H5 files containing 722 clips and
1,444 camera groups; all files opened, the required fields were present, and
the sampled scene-flow values had no non-finite entries.

The matching episode is present in ten clips (`0:11` through `70:81`) with the
two manifest camera serials `20521388` and `24259877`. Each camera has decoded
320x180 RGB, 11-step scene flow, 7-D gripper pose, 7-D joint positions, and
the associated velocity/torque fields. The exact DROID-100 record has 167
trajectory steps and the same camera IDs. Comparing the first RGB frame after
decoding gives MAE 4.46498 (camera 20521388) and 4.46262 (camera 24259877);
the dimensions and camera identity agree, while the JPEG bytes differ because
the two releases use different encoding/processing paths.

This is `PROMISING` for bounded data readiness: an official DROID video record
can be paired with official PointWorld 3-D annotations without acquiring or
using a released model checkpoint. It is not evidence that the native model
benefits from video, because no adapter or training run has been executed.
The complete audit is in
`outputs/cm-pointflow-effect-pretrain/droid100-pointworld-match-20261010-r1/h5_audit_result.json`;
the raw episode metadata is in `source/raw_metadata.json`.

A separate [native-contract Probe](P-20261010-droid100-native-adapter-contract.md)
then checked whether this matched H5 can be passed directly to the current
`H=4,K=24,18-D` learner. It cannot: the released clips have only 11 time points
and do not publish the native point-kind or hand-action tensor contracts. This
does not invalidate the pairing; it bounds what can be claimed from it.

## Next decision

The data contract is concrete enough for separate adapter design, but not for
direct native training. The native-contract audit closes the silent conversion
route; a new route must explicitly choose raw-sequence reprocessing, a shorter
PointWorld-style horizon, or an RGB/robot-state front-end. An RGB-only
front-end would be a different architecture and must not be reported as native
point-flow training. Do not expand to the full DROID corpus or raw MP4/SVO
bucket before one of those contracts is frozen.
