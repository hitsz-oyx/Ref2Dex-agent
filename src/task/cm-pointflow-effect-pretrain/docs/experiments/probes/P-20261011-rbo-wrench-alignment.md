---
schema: ref2dex.probe.v2
probe_id: P-20261011-rbo-wrench-alignment
experiment_id: P-20261011-rbo-wrench-alignment
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: pending
claim_id: C1
hypothesis_family: RBO-object-wrench-adapter
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: retain RBO as a bounded source for point-flow plus geometric-wrench auxiliary training
decision_changed_if_negative: defer RBO and keep only the official-data and simulator-calibrated routes open
status: PLANNED
run_id: rbo-wrench-alignment-20261011-r1
---

# RBO RGB-D / FT alignment Probe

Class: Decision. This Probe tests whether one public RBO interaction archive
has a reproducible, timestamp-aligned contract for RGB-D, object pose, joint
state, and measured force/torque. It does not claim calibrated CHORD
object-COM net wrench supervision.

## Decision note

**Question:** can the public RBO archive be adapted into an auditable geometric
wrench and point-cloud sequence without silently treating sensor wrench as an
object-COM wrench?

**Evidence:** the pinned RBO archive contains 30 Hz RGB-D, about 100 Hz FT and
pose/TF streams, a cardboard-box URDF, and a force sensor URDF. The archive
does not provide payload mass, COM, or inertia, so payload compensation must
remain disabled.

**Cheapest test:** process only the already acquired 42 MB
`cardboardbox26_o` archive. Match every RGB anchor to depth, FT, joint, rigid
body, camera TF, object TF, and force-sensor TF by semantic timestamps within
20 ms. Sample registered depth every 32 pixels, transform it to `/map`, and
transform the raw `myframe` wrench through the sensor URDF and object pose.

## Fixed protocol

- Source: RBO Zenodo record `1036660`, interaction `cardboardbox26_o`; no
  additional archive or checkpoint download.
- Anchor: all timestamped RGB frames in the archive (expected 295).
- Streams: RGB, registered depth, both CameraInfo files, `tf.csv`,
  `rb_poses_array.csv`, `cardboardbox_joint_states.csv`, and
  `ft_sensor_netft_data.csv`.
- Matching: nearest semantic header timestamps, maximum 20 ms; no row-index
  concatenation.
- Wrench layers: preserve `wrench_sensor_myframe`; emit only
  `wrench_object_geometric` with `payload_compensation=false`.
- Point layer: registered depth sampled at stride 32 and transformed through
  the nearest `/map` camera pose. Depth and position units remain explicitly
  marked as inferred meters.
- Output: a new bounded run directory under `outputs/`; source files remain
  read-only and no existing checkpoint is overwritten.

## Results

Pending execution.

## Limitations / future evidence

The archive does not establish payload compensation, object COM net wrench,
cross-object generalization, or downstream Cm benefit. A successful alignment
only justifies a later matched `wrench-on` / `wrench-off` auxiliary Probe. A
failure must be attributed to the adapter or evidence contract, not used to
reject measured-wrench learning in general.
