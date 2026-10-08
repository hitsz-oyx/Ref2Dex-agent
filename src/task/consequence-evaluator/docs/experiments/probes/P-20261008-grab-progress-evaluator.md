---
schema: ref2dex.probe.v2
probe_id: P-20261008-grab-progress-evaluator
experiment_id: P-20261008-grab-progress-evaluator
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: 2be191f
claim_id: C3
hypothesis_family: HF-grab-progress-evaluator
probe_index_in_family: 1
seed_pool: probe
seeds: [229]
decision_changed_if_positive: human-review stage annotations then freeze matched three-seed Oracle comparison
decision_changed_if_negative: inspect label ambiguity, E0 sufficiency and learning before expanding or retraining PointWorld
status: UNCLEAR
run_id: grab-progress-20261008-r1
---

# Ref4: does true object future help learn GRAB grasp progress?

Result: Preparation and implementation pending; no trained progress evaluator or Oracle gain yet.
Decision: First close the native geometry/weak-label interface and run a bounded single-seed Probe; formal three-seed comparison requires human annotation review.

## Authorized scope and decision

User explicitly requests ref4 and chooses automatic weak labels for a small
Probe, followed by human review before formal comparisons. This is a new
independent GRAB mode; existing residual-plan/pair readiness gates, failed
expert runs and continuous episodes remain unchanged. No new branch or push.
It serves the physical-future-to-progress subgoal of C3, not a task success,
long-term Q or robot policy-utility claim.

Decision question: with identical H/A/C, does GT E help predict within-window
grasp stage/progress, and does explicitly derived I improve learnability?
Cheapest evidence: reuse the frozen 30Hz GRAB pack, prepare lift-only weak
sidecars, debug overfit a few train windows then fit three equal-capacity
small models with one common initialization/sample/dropout stream. No fork,
matched preference or six-expert reliability prerequisite in this schema.

## Frozen data/label contract

Reuse PointWorld main-three input manifest's GRAB descriptor and object-level
train/val/test membership. Read raw120Hz motion_intent=lift, original body
contact and table/object mesh/pose; join via saved source_frame_ids, not
SourceWindows' reconstructed frame_ids. Preserve right/left11points/masks,
current-target fixed coordinates and K24/H4. GRAB rotations use v@R, hence
column-vector poses require transposed Rodrigues matrices, including table.
Only retained right-primary/no-other-support fragments with ordered approach,
contact, leave-support and verified-hold boundaries are eligible.

Initial weak-label constants: contact3frames, leave clearance1cm, target3cm,
hold verification15frames; approach wrist displacement1cm, sustained return
descent2cm/3frames. These last motion proposals are explicit engineering
choices, not claims of exact physical intent. Each sequence's tabletop plane
is derived from its mesh and pose; no zero-height assumption. Hold progress
equals1; stop before detected lowering/release. Ambiguous sequences are
excluded with reasons. Save train review SVGs and original-frame boundaries.

Stage proportions alpha are the mean of each training sequence's three
pre-hold duration fractions; holding duration excluded. Validation/test use
the same alpha. Stage/progress/absolute frame/sequence identities are labels
or grouping metadata only. They never enter H/A/C/model inputs. Schema
`ref2dex.grab-progress.v1` has training_allowed=false,
probe_training_allowed=true, annotation_review_status=pending until review.

## Model and matched Probe

Independent geometry PointNet + four history tokens + causal24future tokens,
width128/two Transformer layers/four heads/dropout.1. Current progress is
read at the fourth history token; it cannot depend on candidate future.
Stage4classes/ten-bin soft progress CE, weights1/1; no standalone score head.
Score=mean24future expected progress minus current expected progress.
Shared slots/parameters/normalization across E0/Oracle-E/Oracle-EI. Normalize
with train-only statistics then clear unused future channels. A24x22x9;
target rigid E12 and I22x3=R.T@(P-d). Current support plane/scale/clearance
context shared; future-derived geometry stays in E/I. I is deterministic A/E
re-expression, not an independent contact observation.

Rewind probability.25 only in training, rebuild reversed full28frame geometry,
displacements, history, E/I and context; reverse original labels and mark
augmentation. Never train PointWorld on it or call it a physical failure.
Probe: seed229, batch64, AdamW3e-4/decay1e-4, clipping1,200warmup,
at most1000updates per arm/1800seconds total, validation every250; common
training stream and reset dropout RNG each arm. Engineering overfit uses
debug17, at most100updates on16fixed train windows, no generalization claim.
The ref4 formal10000updates/three initialization seeds remains deferred until
review; neither weak-label loss nor this Probe authorizes formal conclusions.

Report sequence-then-object-balanced progress MAE, per-stage MAE, stage macro
F1, progress-derived score error and separate rewind behavior. Cheap controls:
current geometry progress and history-based extrapolation. Split identities
are never model features; overlap within a sequence is not independent data.
Formal sequence-cluster intervals and matched3seeds follow label review.

## PointWorld and execution bounds

Preselect main-three best.pt46000 by its existing validation, before new
evaluation; freeze SHA and embedded input/output buffers. Predicted path,
if Oracle Probe provides useful signal, retains evaluator weights and replaces
both E and derived I through target_slot. Never mix predicted E with GT I.
Record actual physical errors at8/16/24. Robot q_plan/FK/calibration and
controller gains are later stages, not outputs of this GRAB Probe.

Outputs: `outputs/consequence-evaluator/grab-progress-labels-20261008-r1/`
and `outputs/consequence-evaluator/grab-progress-20261008-r1/`.
Preparation CPU <=600seconds/.5GiB; model work on an afresh empty GPU1,
<=1800seconds/2GiB, shared4GPU/300GB cap and20GiB free reserve.
Stop on frozen input drift, nonfinite values, invalid alignment/split/mask,
deadline, disk cap or foreign GPU use. Commit code before each model run,
record actual SHA/PIDs/device/memory/utilization/ETA and retain failures.

## Literature boundary

[Primary-source notes](../../research/GRAB_PROGRESS_REF4_SOURCES.md): SARM
stage normalization, ReWiND causal reward/rewind and DexWM joint-plan/FK
interface are architectural ingredients. Small geometry inputs, ten bins,
weak-label thresholds and this compute recipe are ref4-specific adaptations.
