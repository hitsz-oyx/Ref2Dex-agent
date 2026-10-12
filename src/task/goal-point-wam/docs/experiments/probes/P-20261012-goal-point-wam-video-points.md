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

Limitations/future evidence: per-frame depth drift, occlusion/scale stability,
moving/static point EPE, source-balanced held-session metrics and target-goal
admission; no causal method-quality, robot or RL claim from smoke.
