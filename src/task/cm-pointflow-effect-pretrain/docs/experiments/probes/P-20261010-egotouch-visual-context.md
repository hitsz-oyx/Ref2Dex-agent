---
schema: ref2dex.probe.v2
probe_id: P-20261010-egotouch-visual-context
experiment_id: P-20261010-egotouch-visual-context
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: pending
claim_id: C1
hypothesis_family: HF-egotouch-visual-context
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: test current visual interaction context as an optional representation teacher input
decision_changed_if_negative: resolve visibility and source correspondence before any visual sensor learner
status: UNCLEAR
run_id: egotouch-diverse-rgb-20261010-r1
---

# Paired visual context after the shape-only teacher screen

Class Decision. Supports optional video/sensor representation pretraining for Cm;
not object-flow supervision, hardware pressure calibration or final robot utility.
The preceding [sensor screen](P-20261010-rawsensor-hand-conditioning.md) is negative
without a gate-changing implementation bug. Its wrist-relative hand shape drops
object state, global wrist transport and contact context. Before another model,
check whether the original paired RGB actually shows useful interaction context.

Fixed protocol before execution: acquire exactly ten chest.mp4 files for the
already selected diverse original records. Preserve their official TRAIN identity
and prior fit/held-task split; no reselection by video content, sensor values or
pseudo-hand availability. Same release cfdbb0ac31cc2af4247943820aa250575e7e6637,
verify original inventory Git/LFS identity and declared bytes. Inventory total
38857473B; declared transfer cap48MiB, per-video12MiB, total acquisition/decode
180s, output cap64MiB. Prior same-release ModelScope discovery found no release;
try domestic HF mirror then existing project proxy, recording failures. No full
corpus/model downloads, external changes or new branch/push.

CPU decoding/statistics only. Compare every original video frame count, frame IDs,
fps and relative PTS to original sensor timestamps. Count distinct decoded frames;
never claim physical synchronization from row/clock agreement. Fixed snapshot IDs
floor((T-1)*q), q=0.1/0.5/0.9, independent of contact/hand/sensor labels. Inspect a
contact sheet for qualitative hand/object visibility, occlusion and contextual
variation; not ground-truth contact labels, no accuracy estimate from 30 frames.
Retain source hashes, timing, fixed snapshots and failure manifest.

Advance only if frame identity is consistent and the fixed views visibly contain
hand-object interaction context in at least3 fit and3 held tasks. This is a data
readiness gate, not a positive learning result. Otherwise do not start visual sensor
learning. Even a passed gate needs a new fixed matched benefit screen before any
native auxiliary integration or large pretraining. Reuse established main3 native
checkpoint; keep geometry/video and raw sensor supervision contracts distinct.

Output: outputs/cm-pointflow-effect-pretrain/egotouch-diverse-rgb-20261010-r1/.
Tool: tools/audit/sample_egotouch_rgb_clock.py with declared budget flags and
--contact-sheet. Stop on source drift, byte/deadline breach or missing pair;
retain verified originals and mark failure. No GPU/model computation at this stage.

## Results

Pending acquisition and root visual inspection.

## Limitations / future evidence

Ten cost-selected TRAIN recordings, previously exposed development tasks, unknown
participant separation. Thirty fixed frames cannot qualify contact localization,
projection/calibration, force labels or temporal causal access. Video appearance
and human future motions are not proposed robot action interventions. Any later
learner must use current/history visual frames only and label-independent window
selection; separate past-only availability from oracle hand visibility masks.
