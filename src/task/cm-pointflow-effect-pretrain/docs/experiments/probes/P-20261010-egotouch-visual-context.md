---
schema: ref2dex.probe.v2
probe_id: P-20261010-egotouch-visual-context
experiment_id: P-20261010-egotouch-visual-context
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 46c787d
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

Acquisition at46c787d completed49.33s,38857473 verified bytes, ten videos.
Every decoded frame is distinct and video/pressure original frame IDs/counts agree;
30Hz,224..751 frames. Max relative PTS deviation0.667ms (engineering consistency,
not physical synchronization proof). Eight records640x480, fruit/pliers320x240.

Root inspected the fixed30-frame contact sheet. All four supported fit and all
five held tasks visibly contain hands and task objects in some snapshots; ball,
fruit, pliers are small/partly occluded. Plush visible but has no complete-hand
training windows and remains unsampled. Readiness gate passes for bounded visual
features, no positive learning claim. Category labels are not physical scene
partitions: Home,Office,Outdoor and heat-gun views share the same table; fruit/
pliers share another background. Only shop-snacks shows a distinct store view.
Do not claim five independent environments or participant independence.

## Limitations / future evidence

Ten cost-selected TRAIN recordings, previously exposed development tasks, unknown
participant separation. Thirty fixed frames cannot qualify contact localization,
projection/calibration, force labels or temporal causal access. Video appearance
and human future motions are not proposed robot action interventions. Any later
learner must use current/history visual frames only and label-independent window
selection; separate past-only availability from oracle hand visibility masks.


## Fixed current-image benefit screen

Next Decision: can CURRENT visual interaction context improve raw sensor forecast,
and does human future shape add information when that context is present? Two
separate gates avoid calling image-only context improvement an action-conditioned
teacher. No additional data/weights download. Use existing local ImageNet ResNet18
checkpoint resnet18-f37072fd.pth (verify hash prefix), frozen/eval, resize short
side256 and center224 crop with official local torchvision V1 transform. Report
external ImageNet pretraining; it is not learned from our fit set. Extract per-frame
512-dimensional features in source order on one freshly free GPU, BF16, batch64.
Frame-local features do not depend on adjacent/future RGB. Fixed original video/
label hashes, all original frame IDs, no pressure/contact-driven image selection.
Keep all original feature rows; learner reads exactly start+3, never future images.
GPU extraction cap180s, feature output cap16MiB; no encoder updates.

Same330/535 complete windows and seed227. Six identical MLP arms, same initial
weights/task-uniform batches/500updates/optimizer/Huber loss as prior screen:
history; future_hand; history_rgb; future_hand_rgb; shuffled_future_hand_rgb;
shuffled_rgb (actual future shape, different-task current image). All include the
same feature dimensions; withheld portions zero AFTER frozen FIT mean/std.
Shuffle uses next sorted different task/modulo-index, independently within each
split, without target values. Match architecture/capacity, not previous checkpoint
optimization trajectory. RGB and sensor targets have distinct paths and hashes.

Primary observation-context gate: history_rgb groupA h24 held-task macro MAE at
least10% better than best(history,persistence), no held task>20% worse than its best
of those controls. Conditional-human-motion gate: future_hand_rgb at least10%
better than best of all other arms and persistence, with same per-task safeguard.
Use fixed final checkpoint, no dev selection. GroupA/B are normalization-processing
candidates, UNKNOWN hardware roles. PROMISING for at least one passed gate, with
which gate explicit; a context-only success is not a positive future-motion teacher.
Both fail => UNPROMISING for this current-view recipe, not visual/tactile useless.

GPU training/evaluation cap300s and output cap400MiB; same free GPU1 if still free,
else reselect. Log utilization/memory/ETA; stop on conflict/nonfinite/hash drift or
caps. Preserve sources/checkpoints. No native corpus integration, longer fitting,
remote operations or robot-policy training follows automatically from this screen.
