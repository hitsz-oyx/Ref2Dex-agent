---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-video-points
experiment_id: P-20261012-goal-point-wam-video-points
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 23947ec4
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 15
seed_pool: probe
seeds: [287]
decision_changed_if_positive: admit quality-audited video point trajectories into mixed shared-backbone training
decision_changed_if_negative: repair teacher-observation alignment before scaling video supervision
status: UNCLEAR
run_id: video-points-engineering-20261012-r1
---

# Shared backbone and direct video point supervision

Class: Blocker/engineering first. Serves the Task's goal-conditioned hand and
action-conditioned point-transition representation upstream of Mission C3.
User approves point trajectory supervision across datasets, shared actual
backbone parameters, optional SE(3) native head decoded to the common point
loss. Missing video object poses no longer block supervision.

Decision Note: retain the existing action-conditioned world interface for Cm;
add an independent direct point head and explicit video data contract. Native
and video batches use the SAME scene encoder, scene projection, hand/goal
decoder and action adapter. No copied video backbone. Native rigid head stays
optional; both decoders yield24-step metric point positions and common masked
Huber loss. Native R/t auxiliary losses are omitted in this new training entry.
This is explicit weight transfer from final10000, fresh optimizer/new checkpoint
kind, not a resume or a change to the active isolated remote job.

## Cheapest discriminating checks

1. Semantic tests: future labels do not enter observations; point permutation/
   query chunking preserve correspondence; missing/NaN labels contribute no
   gradients; duplicate point density does not inflate a window's loss.
2. Prepare one already acquired official TRAIN clip (no downloads), current
   frame15 and observed native frames12--15. Run pinned Omega on ONLY those
   four prefix frames, calibrate metric scale with confidence-qualified native
   hands. First-frame pixel grid chooses points using observed depth/voxel and
   workspace only. Existing full-clip CoTracker/depth provide FUTURE LABELS.
   Report independent prefix/teacher current-depth disagreement without
   concealing it via label rebasing. Labels have10Hz measurements interpolated
   to30Hz; native2x11 hand labels remain original30Hz.
3. Four-update one-GPU mixed native/video smoke, bounded900s, no checkpoint,
   main batch16/video batch4, max2048 video points, outputs <=20MB including
   prefix images. Check finite gradients and video loss reaching shared PTv3.
   GPU snapshot and ETA in progress log. Optional two-rank DDP wiring check
   uses the same four updates, not new seeds or method comparisons.

Inputs: original local official-TRAIN acquisition
`outputs/cm-pointflow-effect-pretrain/pointwam-train-inputs-20261011-r1/`;
existing qualified `episode00-basic_pick_place-0/clip01-f0015/teacher_frontend.npz`
under `pointwam-train-qualification-20261011-r1`; pinned local teacher assets.
New outputs: `outputs/goal-point-wam/video-points-admission-20261012-r1/` and
`outputs/goal-point-wam/video-points-engineering-20261012-r1/`.

No scientific fit until teacher drift/geometry inspection and separate-session
train/dev admission. Null goals are allowed for video point-world supervision
without inventing target-object IDs; optional current-frame target mask supplies
paired goal points and held-visible future goal masks. Full video object-goal
supervision needs that mask. The current-only video scene has no measured
surface normal/dynamics fields; they are explicitly zeroed, native hand history
is retained. This limitation must be considered during fit, not hidden as ground
truth static scene history.

Result: UNCLEAR; engineering wiring passed; scientific label admission pending.
Decision: retain point-supervision implementation if wiring passes; admit a
scientific video corpus only after label quality checks.

## Engineering and label-quality results

Code23947ec4: all goal Task tests28 passed/1 skipped (9.13s). Four semantic
point tests passed, including masked NaN labels, density normalization,
query permutation/chunking, future/observation isolation and admission rejection.
Observed-prefix preparation completed35.43s using only native frames12--15:
60 confidence-qualified scale samples, metric scale0.619133, IQR0.125474,
2048 observed scene queries and29505 valid future point labels. Native hands
remain11 real measured points, no padded10-point palm substitute. No object
mask was supplied; video goals are explicitly null.

Single GPU0 mixed four-update smoke completed5.61s; reserved2.34GB.
Two-rank GPUs0/1 smoke completed6.56s; rank0 reserved2.55GB. Both video updates
mask actor supervision so nonzero backbone gradient comes specifically from
VIDEO WORLD point loss (9.14e-5 single,8.30e-5 DDP after the zero-initialized
head first update). Final DDP parameter hashes identical:
`eac66e019bc6eb9f07749bad3612a808fb83cfea02728a5a19c345e943801c71`.
These are engineering-only runs, no checkpoints saved.

Review found the first video adapter mistakenly used velocity/acceleration
in future action features. Native action normalization expects
`[absolute XYZ, displacement from current XYZ, one-tick displacement]`, all
in meters. Corrected before scientific admission; added direct numeric tests
of these units. Unknown native local-instance metadata also stays zero.
Original smoke records are preserved as pre-fix engineering evidence; their
numerics do not describe the corrected action contract. Re-run only the
affected mixed smoke after the fix, not scientific training.

Quality audit: independent prefix-current and full-clip teacher depth disagree
by median33.575mm/p90 52.089mm. Current query pixels match teacher identity
to2.16e-5px, so the large offset is not simple query-ID mismatch. Among246
camera-compensated low-image-motion audit candidates, the teacher's own
median world displacement is3.3--6.3mm over0.1--1s; this heuristic is not proof
of staticness. Records: admission `manifest.json` and `label_quality.json`.
Raw absolute video labels therefore need consistent current/teacher geometry
handling and quality inspection before scientific admission. Do not hide the
offset by silently rebasing labels or claim four updates prove method quality.

Corrected-action repeat code69c69de7:
`video-points-ddp-engineering-20261012-r2` completed4 updates in5.97s,
video WORLD-only backbone gradient9.91e-5, finite gradients and equal two-rank
final parameters `c3b61e9e35c2326d464ffcde261d7b2dc03c994589c2f340b221ea61be3abdf6`.
Reserved2.55GB on rank0; no checkpoint, no scientific fit. Explicit action-unit
semantic tests again4 passed. User subsequently prioritizesref2 robot integration;
video label-quality work deferred while preserving this reusable interface.

Limitations/future evidence: per-frame depth drift, occlusion/scale stability,
moving/static point EPE, source-balanced held-session metrics and target-goal
admission; no causal method-quality, robot or RL claim from smoke.

## Video-agent takeover and rolling-depth Decision (2026-10-12)

User explicitly assigns video/EgoDex work to this separate session; robot
integration stays with traj_policy. Existing remote native auxiliary training
is preserved and monitored, not mislabeled as scene/video supervision. At7770
updates it uses all four remote GPUs; main macro44.626mm and EgoDex held
macro50.172mm at8000 are monitoring observations, not matched causal evidence.
The existing remote guard and300GB global storage limit remain binding.

Decision question: can using one identical observed-four-frame depth frontend
for current inputs and future label frames remove cross-frontend alignment
error without introducing unacceptable temporal drift? Old absolute teacher
alignment median33.575/p9052.089mm; a diagnostic global depth ratio.964063
reduces median to9.854mm but leaves local shape differences. No label offsets
are subtracted, and this fitted ratio does not enter input or labels.

Root selects a bounded engineering comparison, then a mixed learning Probe
only if label screening passes. `--label-depth rolling-prefix` reuses original
CoTracker pixel identities/visibility and native cameras, but evaluates Omega
once per sampled label tick on that tick's four consecutive native past/current
frames, with native2x11/confidence>=.5 hand-depth calibration. Each label frame
can use later video than the actor current time solely as supervision. The
first depth result itself supplies current input, so frontend identity is exact
without per-track rebasing. Nine10Hz measurements span0--.8s, interpolated to
25 positions;28 native hand frames remain H4/K24. Future camera transforms
participate only in label lifting into the single current frame.

Before seeing new results, screening requires current alignment p90<=.25mm,
>=32 camera-compensated low-image-motion background candidates, their largest
timewise median displacement<=10mm/p90<=25mm, all scale-calibration relative
IQR<=.35,>=256 observed queries and>=25% future-label availability. Background
candidates are audit strata, never future-based input point selection. Passing
this heuristic is permission for an exploratory pseudo-label fit, not proof of
ground-truth depth/motion. Record original full-teacher discrepancy as well.
Failure keeps original admission flags false and leads to a geometry repair;
do not automatically relax thresholds or declare EgoDex ineffective.

Reuse existing24 native-admitted official-TRAIN records, native session split283
and completed acquisition bytes. Start with basic_pick_place0/1 (train/val),
then screen remaining existing nonzero-start clips if those pass. No archive
download, new branch, full-corpus claim, extra seed/model controls or object
SE(3) prerequisite. Null video object goals remain explicit; original main-three
batches continue object-goal/rigid-point supervision on the shared backbone.

Remote processing starts after original four-rank job completes and the assigned
GPU has no other job; existing sources/checkpoints are never modified. First
two clips<=900s each,<=20MB/clip; expansion<=24 clips/480MB and<=1h wall on
at most four remote GPUs. New sparse code checkout/data/logs remain in the
existing authorized isolated NAS scope. Engineering manifests stay inadmissible
until screen and frozen session split are bound to a new aggregate manifest.
All original failed/successful raw and teacher records remain untouched.

Point trainer now monitors a fixed64/source subset of the existing frozen
main panel, cached true/static goal response, and held-session video point EPE;
records rank sampling, actual parent step/hash, fresh optimizer and new model
kind. Periodic atomic latest includes optimizer and per-rank RNG; final uses a
hardlink. All ranks enter the same evaluation/save barriers, independent of
whether a rank writes metrics. Actual GPU preflight/label results are pending.
CPU scheduler/point semantic tests6 passed; CPU label audit reproduces failure
only at the old current-alignment gate. Model computation uses GPU; CPU only
performs these array/contract checks.

Runtime takeover and diagnostics: `tmp/video-training-20261012-takeover/`.
Planned remote artifacts: isolated NAS scope `video-repo/` and
`video-points-20261012-r1/`; original EgoDex r2 remains separate. Budget must
be recomputed including atomic save peaks before any scientific fit. Stop on
identity drift, nonfinite/OOM, GPU conflict, storage threshold or deadline.
