---
schema: ref2dex.probe.v2
probe_id: P-20261010-droid100-raw-input-package-audit
experiment_id: P-20261010-droid100-raw-input-package-audit
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: ee86930
claim_id: C1
hypothesis_family: official-raw-video-data
probe_index_in_family: 3
seed_pool: probe
seeds: []
decision_changed_if_positive: stage the official raw PointWorld processor for one episode
decision_changed_if_negative: do not install system ZED dependencies or fabricate camera/depth fields
requested_intervention: official_video_data_training
executed_intervention: official_droid_raw_input_package_audit
status: UNCLEAR
run_id: droid100-raw-input-package-20261010-r1
---

# Does the raw DROID episode already contain the inputs needed for 3-D reprocessing?

Class: Decision. The raw MP4 temporal audit showed enough frames for the native
horizon. This read-only audit checks the companion HDF5 files before attempting
the official PointWorld processor; it does not install system packages, run
ZED, download SVO files, or train a model.

## Result

The official `trajectory.h5` object is 1,069,889 bytes with SHA256
`a78cb687680a9fcc9fa78f7355d00203cd08173a118936d72f306657321ba864`. It has
167-row robot state/action arrays, camera and robot timestamps, and 6-D camera
extrinsic records. The official `trajectory_im128.h5` object is 81,582,742
bytes with SHA256
`59fc0c1feb97aaec21131968f4d86253cc12799ac8eb5bb38bbdec91f3f4e132`; it has
154 rows, six 128x128 RGB streams and 4x4 camera extrinsics. Neither audited
file exposes a depth dataset or camera intrinsic matrix.

The pinned PointWorld data branch's `real/droid_utils.py` reads the DROID SVO
files through `pyzed` to obtain stereo frames, calibration and optional depth.
This environment has no `pyzed`, ZED SDK library, CoTracker, FoundationStereo
or VGGT installation. Installing a system ZED SDK would cross the repository's
protected system-modification boundary, and inferring intrinsics from the MP4
would introduce an unverified geometry assumption. The result is therefore
`UNCLEAR` for running the official 3-D reprocessing route in this environment,
even though the raw episode itself is available.

## Next decision

Keep the official raw files and H5 annotations read-only. A valid raw
reprocessing Probe requires either an already authorized environment containing
the pinned ZED/vision dependencies or a user-approved, reproducible replacement
for SVO calibration/depth. Until that boundary is resolved, do not start
training or silently substitute 2-D/RGB fields for the native 3-D contract.
