---
schema: ref2dex.probe.v2
probe_id: P-20261010-egotouch-label-schema
experiment_id: P-20261010-egotouch-label-schema
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: a33305e905042c77bae6f5c8823ffe6663b992c7
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

Result: UNCLEAR for training readiness. Acquisition completed in129.97s, with
485754 bytes and10 verified files. Domestic mirror requests timed out; official
HF through the project proxy succeeded. Every size and Git/LFS checksum matched
the pinned revision. No video or model was downloaded.

Acquisition: `manifest.json` under the run output. Numeric audit:
`label-audit.json`, executed at `f8ff894` using
`tools/audit/audit_egotouch_label_schema.py`.

| Sample (official TRAIN) | USB cable | mouse |
| --- | --- | --- |
| Task / recording |Home/plug_unplug_usb_cable/20260320_135743_682|Office/slide_mouse/20260314_161602_832|
| Pressure / Wilor / Vive rows |43 /43 /43|47 /47 /47|
| Source frame IDs |all three match, contiguous|all three match, contiguous|
| Pressure clock step range |33.3333–33.3335ms|33.0000–34.0002ms|
| Wilor clock |relative, synthetic30Hz|relative, synthetic30Hz|
| Pressure / Vive timestamps |exactly match|exactly match|
| Valid cells /441, per hand |217|217|
| Per-record tactile / bend max |20 /30|50 /45|
| Baseline corrected / separate normalization |true /true|true /true|
| Wilor `aligned_to_vive` |false|false|
| Explicit joint confidence / validity |absent|absent|
| Camera calibration file |absent|absent|
| Clip-level annotated contact (left / right) |false /false|false /false|

Unmeasured cells are NaN with a constant spatial mask, not observed zero
pressure. Both 21-joint hands are finite, but original z values range6.21–19.07
in the released coordinate convention; no calibrated stationary-world or
physical-unit claim follows. Equal frame IDs are an engineering alignment
signal; RGB synchronization remains untested. Both samples contain false/false annotation values. The semantics and
completeness of those flags have not been qualified, so physical absence of
contact cannot be inferred from them.
Normalized nonzero cells do not establish contact or calibrated force.

Decision: keep tactile training gated. Next cheap tactile probe should select
a verified active-pressure TRAIN episode, qualify the annotation semantics, retain original
frame IDs and sparse sensor masks, separate pressure from bend, and establish
RGB/camera relations before hand-world targets. Preserve per-record maxima;
these grids are not directly comparable calibrated force targets. Tactile
remains an optional auxiliary target/teacher for the geometry route, not a
required robot inference input.

## Follow-up protocol: contact-positive sample (before execution)

Run `egotouch-contact-positive-20261010-r2` continues this same schema question.
Search at most40 distinct original TRAIN tasks for a true original left/right
contact annotation, favoring manipulation task names and then smallest complete
bundles. Task name alone never establishes contact. Fixed raw revision and
Git/LFS checks remain unchanged. Download one positive bundle's five labels and
chest RGB only; total transfer<=16MiB, wall<=300s, new local directory. Reuse
ModelScope discovery; try mirror first, then project proxy, retaining the failed
mirror result for remaining files. Decode chest frame count/fps and compare
original frame-index/timestamp/pressure layout in a numeric audit. Equal counts
are insufficient to prove synchronization; no camera calibration is inferred.
Tool: `tools/audit/sample_egotouch_contact_positive.py`. This is CPU acquisition/
schema checking, not model training or a change in the Mission claim.

## Follow-up results and correction

The bounded search at `7a0dc86` completed in28.10s, transferring10934 bytes.
All40 distinct TRAIN tasks had false/false contact flags; no RGB was acquired.
This includes names suggesting grasp/squeeze, but task names are not physical
contact evidence either. Preserve `egotouch-contact-positive-20261010-r2/manifest.json`
and all original annotations. These flags cannot currently serve as a reliable
positive/negative training filter. Earlier wording implying physical no-contact
from false flags is corrected above. The release converter examined so far does
not establish the annotation provenance or coverage; default/unvalidated flags
are a hypothesis, not a finding.

Next tactile decision: audit pressure-channel roles/normalization and temporal
activation in the existing two verified records, then acquire one small chest
RGB to check original frame identity. Pressure activation is not force calibration
or contact GT. Keep tactile as an optional auxiliary teacher; do not block the
video-only geometry route on annotation availability.

## Limitations / future evidence

Pressure values do not establish Pa/N calibration. Sparse 21x21 mapped values
are not441 independent sensor readings. Wilor hand estimates are pseudo-labels.
No object trajectory or camera extrinsic calibration is inferred from Vive
tracker records alone. Without RGB, equal label-array counts cannot establish
video synchronization or useful point-dynamics supervision. No main-source
training registration or deployment tactile-input requirement is introduced.

Before the RGB-clock follow-up: acquire only `chest.mp4` of the two existing
verified TRAIN records (306893 and62916 bytes in the pinned inventory). Reuse
ModelScope discovery; mirror first then project proxy. Verify original size and
Git/LFS identity, cap2MiB transfer/90s, no annotation-positive assertion. Decode
RGB and extract video PTS with ffprobe; compare frame-index/count and relative
pressure times. Count distinct decoded frames and report normalized grid temporal
changes without interpreting the mixed tactile/bend grid as contact or force.
Output `egotouch-rgb-clock-20261010-r3/`; CPU file/video audit only. Agreement is
engineering consistency, not proof of physical synchronization or calibration.
