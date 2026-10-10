---
schema: ref2dex.probe.v2
probe_id: P-20261010-egotouch-label-schema
experiment_id: P-20261010-egotouch-label-schema
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: pending
claim_id: C1
hypothesis_family: HF-tactile-data-readiness
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: test original frame-index alignment and pressure masks in a bounded tactile adapter
decision_changed_if_negative: defer tactile training until labels and clocks can be qualified
status: UNCLEAR
run_id: egotouch-label-schema-20261010-r1
---

# EgoTouch original label schema sample

Class: Decision, CPU file/label audit, no model training.

Question: Are the released pressure arrays, original Wilor frame identities and
pose/annotation clocks sufficiently specified to attempt a tactile auxiliary
target adapter? This serves point-dynamics representation pretraining, without
changing C1 or asserting tactile-to-robot force calibration.

## Pre-run decision note and protocol

Choose a cheap schema sample rather than a large RGB acquisition: two distinct
TRAIN tasks, selected by smallest combined label size among recordings present
in the official split and raw inventory. This is explicitly selection-biased,
not a quality estimate for the full corpus. Download only five files per record:
`jq_pressure.json`, `pressure_grids.npz`, `wilor_hands.json`, `vive_poses.json`,
`manual_contact_annotation.json`. No videos, models or HDF5 bulk conversion.

Pinned raw dataset: `zhouzhoujy/EgoTouch` revision
`cfdbb0ac31cc2af4247943820aa250575e7e6637`; verify every byte count and Git blob
or LFS SHA against the previously pinned complete inventory. ModelScope search
on 2026-10-10 found no matching EgoTouch release (search absence is not proof of
absence). Try the domestic HF mirror first, then official HF via the repository
proxy. Cap total transferred bytes at16MiB and acquisition wall time at180s;
stop on checksum mismatch, missing split authority, or cap. Partial verified
files and failed-attempt manifests remain in the new output directory.

Output: `outputs/cm-pointflow-effect-pretrain/egotouch-label-schema-20261010-r1/`.
Tool: `tools/audit/sample_egotouch_labels.py` relative to this Task. Inventory and
split snapshots are in `tmp/video-tactile-primary-sources/` of this worktree;
their hashes are recorded in the run manifest. CPU is used for JSON/NPZ schema
and numeric checks only. No GPU or native corpus/checkpoint modification.

Positive: retain original frame numbers and sensor validity for a small adapter
probe before any training. Negative/ambiguous: preserve sample and resolve
missing clocks/calibration first. No new external authorization is needed for
this bounded public-data sample under the user's video/tactile exploration.

## Results

Pending execution of the committed protocol.

## Limitations / future evidence

Pressure values do not establish Pa/N calibration. Sparse 21x21 mapped values
are not441 independent sensor readings. Wilor hand estimates are pseudo-labels.
No object trajectory or camera extrinsic calibration is inferred from Vive
tracker records alone. Without RGB, equal label-array counts cannot establish
video synchronization or useful point-dynamics supervision. No main-source
training registration or deployment tactile-input requirement is introduced.
