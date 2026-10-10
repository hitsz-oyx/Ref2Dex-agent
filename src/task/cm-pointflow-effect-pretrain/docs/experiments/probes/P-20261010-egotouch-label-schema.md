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


RGB clock audit completed at `c57dcbf`,6.25s,369809 verified bytes. Both videos
are30Hz and contain43/47 distinct decoded frames, matching contiguous original
pressure IDs. Maximum relative PTS/pressure-clock deviation is0.445microseconds
for USB cable and0.333ms for mouse. This qualifies row/relative-clock consistency
for these two samples only, not sensor latency, physical synchronization or
world calibration. Per-hand normalized-grid mean frame changes range0.00306–0.01309;
nonzero values alone remain uninterpretable as contact before bend separation.
Both raw videos and full original PTS are preserved in the follow-up manifest.

Root independently counted the official fixed mappings:217 display cells map to
138 distinct raw indices on the left and147 on the right; a raw value can alias
to5 cells. The cell count is not an independent sensor count. See the preserved
`normalization-mapping-audit.json` with mapping/grid hashes. Both actual NPZs
store global `tactile_max`/`bend_max`, while the fixed release HDF5 converter
reads per-hand `*_max_left/right` with0 defaults; those four keys are absent.
Do not silently convert the metadata to zero or infer separate hand maxima.

The follow-up channel investigation can numerically reproduce these two grids
with candidate normalization groups, but hardware tactile/bend semantics remain
unconfirmed. Whole-record maximum normalization can depend on future frames;
future adapters must use raw/fixed-scale or TRAIN-only statistics for historical
sensor inputs. Preserve actual metadata and qualify any target normalization
separately. No pressure-only mask or calibrated-force training claim is made.


The [channel-contract note](../../research/2026-10-10-egotouch-tactile-channel-contract.md)
contains candidate normalization groups and right-hand spatial repair. Root
executed its read-only reconstruction on both original records: all NaN masks
match and maximum cell errors are1.19e-8/2.78e-8 and9.54e-9/2.91e-8 (left/right).
This verifies sample-level numerical reconstruction, not hardware semantic roles
or full-release consistency. Keep those roles UNKNOWN; the next engineering
adapter can predict original sensors with raw/255 and explicit alias/imputation
provenance without asserting calibrated contact or pressure.

## Raw sensor dynamics audit (before execution)

Class Decision: determine if existing raw bytes offer nontrivial temporal
supervision without processed-grid future normalization or inferred contact.
Use exactly two verified records; sorted USB is fitting material and mouse is
held-record development (both originally official TRAIN). Keep original frame
IDs/actual sensor timestamps, H4/K8. Scale raw512 bimanual channels by fixed255;
no baseline subtraction, record maximum, grid aliases or bend-role interpretation.
Freeze changing channels from fitting record raw range>=2 counts, requiring>=16
channels. This uses fitting labels only; report all channels and frozen-active
ones separately so missing newly active held channels remain visible.

Compare persistence, last-two CV and four-history OLS at actual elapsed time,
clipped only to the known byte range. No model training/GPU. Report MAE all/h8,
left/right, support, and fraction changing>=2 counts. If held-record OLS h8 MAE
beats persistence by>=10%, raw temporal predictability justifies a bounded
sensor-representation probe; otherwise use manipulation conditioning/diverse
label acquisition as the next question. This is raw sensor prediction, not force,
contact or world-point supervision. CPU<=30s, <=100KiB, output
`raw-dynamics-audit.json` under the original acquisition; original files untouched.


The fixed-scale two-record raw dynamics audit at `5bee4df` finds65 changing
FIT channels versus137 in held mouse. h8 active-channel normalized MAE for
persistence/OLS is.013625/.021940 on held mouse; no preset history signal.
Fitting USB has only8.13% h8 active point-channels changing>=2 byte counts.
These short samples were inadequate for a learning decision. Follow-up longer
[raw sensor/hand conditioning screen](P-20261010-rawsensor-hand-conditioning.md)
acquired10 original TRAIN tasks and preserves missing WiLoR hands explicitly.
