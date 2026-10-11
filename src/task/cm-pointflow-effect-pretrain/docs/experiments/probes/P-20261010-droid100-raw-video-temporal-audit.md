---
schema: ref2dex.probe.v2
probe_id: P-20261010-droid100-raw-video-temporal-audit
experiment_id: P-20261010-droid100-raw-video-temporal-audit
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: c49c088
claim_id: C1
hypothesis_family: official-raw-video-data
probe_index_in_family: 2
seed_pool: probe
seeds: []
decision_changed_if_positive: evaluate the official PointWorld raw-data pipeline on one bounded episode
decision_changed_if_negative: keep raw-video reprocessing closed and retain only the RLDS/data audit
requested_intervention: official_video_data_training
executed_intervention: official_droid_raw_mp4_temporal_audit
status: PROMISING
run_id: droid100-raw-video-temporal-20261010-r1
---

# Does the matched official raw DROID episode have native temporal coverage?

Class: Decision. The H5 pairing is real but its released clips contain only 11
time points. This read-only audit checks one official raw MP4 from the same
episode, so that a possible longer-sequence route is distinguished from the
short H5 contract. It does not download the stereo partner/SVO, run PointWorld
processing, train a model, or use a released checkpoint.

## Result

The GCS object is
`robotics/droid_raw/1.0.1/RAIL/success/2023-04-17/Mon_Apr_17_14:48:05_2023/recordings/MP4/20521388.mp4`.
The local 6,082,163-byte copy matches the object metadata MD5
`6rbged7tHZeSOvK2JKvc1A==` and has SHA256
`ba3581cddd3ca3b54737daa5079063fa16a03c7b2fe4f6d2bdb6d1daaf80fe6f`.
`ffprobe` reports H.264, 1280x720, 60 FPS, 166 frames and 2.767 seconds. This
is enough raw temporal coverage for 4 history plus 24 future frames; the H5
11-frame limit is a PointWorld processing choice, not a source-video limit.

After resizing raw frame 0 to the H5 320x180 contract, the pixel MAE/RMSE to
the matching PointWorld initial RGB are 4.36365/5.49408. The dimensions and
camera identity agree, while the different encoding/processing paths prevent a
byte-identity claim. The full probe JSON is the ignored artifact
`tmp/droid100-raw-video-audit-20261010.json`.

This is `PROMISING` for raw temporal availability. It does not establish that
longer raw clips can be lifted to the native 18-D point-flow contract.

## Next decision

The official PointWorld data branch exposes `--frames_per_clip`; its default is
11, so one bounded reprocessing Probe could test 28-frame clips on this
episode. That route requires both external cameras, trajectory metadata,
stereo depth, camera extrinsics and the released CoTracker/FoundationStereo/
VGGT checkpoints. It must also define point identity, point-kind labels and
native hand/action fields before training. Do not silently convert the raw MP4
into native samples or scale to more episodes until that full contract is
frozen.
