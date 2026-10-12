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

## Clarification: offline human preprocessing versus online observations

The user's PointWAM question prompted a primary-source recheck of
[Appendix B.1](https://arxiv.org/html/2610.02840v1#A2.SS1): its human-data
pipeline uses offline clip reconstruction, clip-level hand scale calibration
and offline tracks; it does not certify prefix-only depth for human training.
Our four-frame causal observation requirement was an additional adaptation,
not a paper requirement. Combining independently reconstructed prefix inputs
with full-clip absolute labels introduced the measured33.575mm median offset.
That is an implementation alignment blocker, not evidence against EgoDex or
the paper's method. The existing remote run is hand-only auxiliary training,
not the paper's joint human scene/hand training.

Decision Note: first screen same-reconstruction offline human pretraining.
Use the existing teacher's initial depth for current queries and its future
depth for labels; preserve native cameras, grid identities and all original
arrays. This tests whether original teacher quality, rather than incompatible
frontends, blocks a mixed Probe. No global fitted scale or per-track rebasing.
This replaces the rolling-depth comparison as the immediate next step; its
code remains available but no rolling-depth remote job has been started.
The new mode is explicitly `offline-teacher`, `causal_observation=false`,
`observation_policy=offline_clip_reconstruction`, `offline_pretraining_only=true`.
Loader accepts it only with those explicit declarations; engineering data
still cannot enter a scientific fit. It cannot establish online causality,
deployable perception quality or a formal video transfer claim.

Cheapest check: CPU-only reuse of already acquired depth/tracks for the same
basic_pick_place train/dev pair,<=900s/clip and<=20MB/clip. No neural model
computation and no GPU allocation are needed. Preserve the existing quality
screen, inspect the results, then decide whether remaining22 clips and a GPU
mixed smoke are useful. Stop on source drift, geometry-screen failure or scope
budget/deadline. A new scientific fit remains conditional on separate-session
admission and GPU availability after the original run finishes. Existing main
architecture and Cm interface remain the Task adaptation, not an official
PointWAM reproduction; the author project page still shows Code soon.

Offline screen code `d497685d`, owned remote driver PID208763: both clips were
prepared without model/GPU computation and retained `training_allowed=false`.
basic_pick_place0 current-alignment median0/p90.000169mm,244 background audit
candidates, maximum median6.604mm/p9013.046mm; all six geometry screens pass.
Held-session basic_pick_place1 current alignment median0/p90.000239mm, but
only15 background candidates, maximum median16.548mm/p9028.570mm; three
geometry screens fail. Exact shared geometry removes the cross-frontend offset
in both records, but does not make reconstructed depth or tracks ground truth.
The pair is therefore not scientifically admitted. Neither a mixed video fit
nor a rolling-depth job was launched. Next cheapest decision is to determine
whether the dev failure is clip-specific or shared teacher geometry drift,
using existing frozen-dev clips and source calibration/visualization evidence.
Do not infer method ineffectiveness or change the frozen recording-session split.
Receipt and measured quality: `tmp/video-training-20261012-takeover/`; remote
`video-offline-screen-20261012-r1/`. Tests7 passed, including explicit offline
scope acceptance and retained unadmitted-data rejection.

## Cached-trajectory extraction decision

User explicitly requests preparing the complete trajectory once, then extracting
windows. The offline-teacher path implements that contract per already processed
one-second clip; long episodes may stay partitioned into bounded clips, with
clip-local track identities. Cache is read-only. Native30Hz offsets0/3/6 each
have H4 measured hand history and K24 future labels; later windows use current
tracked pixels/visibility and the same cached metric depth for both current
queries and future labels. Future visibility or motion does not select current
queries. Native current camera anchors each window, without rerunning Omega
or recalibrating scale. Copy existing calibration provenance and verify its
source clock/hash where available; do not replace it with a new estimate.

Decision: inspect existing frozen-dev clips at all three offsets to determine
whether the first dev failure is local or shared. Then extract existing train
clips only if dev geometry offers a viable Probe. CPU-only extraction/audit,
<=1h and <=480MB total, no new download or GPU inference. Quality failures stay
excluded, frozen session split is preserved, and aggregate admission must record
all exclusions. A GPU mixed smoke can use engineering data after the old four-rank
run and GPUs have fully exited; scientific fit still needs both admitted splits,
fresh total storage accounting and explicit Probe identity. This is a reuse and
window-extraction change, not a PointWAM architecture or global-claim change.

## Cached-window results and first mixed learning Probe

Code9c81eaa2, remote driver212312 COMPLETED: native offsets0/3/6 of train
basic_pick_place0 pass all geometry and recovered-scale screens; frozen-dev
basic_pick_place1 offset6 passes, offsets0/3 fail and remain excluded. This is
four admitted-quality windows from two sessions, not new unique episodes.
Independent HDF5 checks verify current/final native hand ticks and the next
10Hz scene measurement at all three train offsets: maximum coordinate residual
2.19e-7m. Current alignment is0 for later windows. Cached source depths/tracks
were not recomputed or modified. The original insufficient dev0 window remains
preserved; the changed window contains a different, explicitly recorded interval.

Actual one-GPU four-update mixed smoke completed2.39s, losses/gradients finite,
video WORLD-only shared-backbone gradient4.17e-5; reserved2.50GB, physical2.88GB.
Native auxiliary final10000 checkpoint transfer hash
`5ea08456ecb77125de557c19d2009b49265f66e24678eed82d448a5029a27922`;
new point head/fresh optimizer, no smoke checkpoint. Existing projection/depth
views are inspected as rough geometry evidence, not depth/motion ground truth.

Decision Note: can a tiny same-reconstruction, action-conditioned video fit
improve held-session point trajectories while retaining the native WAM? Run
300 mixed updates (3 native:1 video), seed287/draw288, one free remote GPU,
native batch16/video4, lr1e-4,<=900s. Separate immutable aggregate admission
manifest records the three train/one dev windows, source hashes, seven quality
gates and both excluded dev windows; do not mutate engineering manifests.
Null video goals stay explicit. Fixed main64/source panel and cached true/static
goal response are evaluated before and after. Only final checkpoint is saved
(periodic intervals1000); latest/final hardlinked, fresh optimizer/RNG retained.

Before fitting, positive screening is held-video point EPE >=5% lower than its
initial/static baseline, main hand macro <=105% and main point macro <=110%
of this run's initial panel. Positive permits wider existing-data extraction;
otherwise inspect fit/capacity/noise and use UNCLEAR with this tiny sample,
not a negative method claim. No official PointWAM, formal transfer or RL claim.

Root adjusts only the isolated NAS self-allocation to6.0GB for new checkpoint
storage; Campaign300GB and combined298GB proactive stop remain unchanged.
Measured pre-fit combined artifacts~296.54GB; one final checkpoint estimated
<=0.70GB, <=1.40GB conservative allowance for any atomic replacement. Stop on
drift/nonfinite/OOM/GPU conflict, deadline or storage threshold. Never overwrite
original auxiliary final or engineering arrays. New run_id=
video-offline-mixed-20261012-r1, isolated NAS scope/video-offline-mixed-20261012-r1.

First launch r1 failed before creating a training process: temporary admission
assembly assigned a dictionary to the supervisor's `record` function name,
so writing TRAINING status raised TypeError. No model update, GPU allocation
or checkpoint occurred. Preserve r1 admission/logs. Rename the row variable,
assert the writer remains callable, and retry identical protocol as
video-offline-mixed-20261012-r2; do not reinterpret this launcher defect as a
learning result. Core training code stays pinned9c81eaa2.

Retry r2 also exited in CPU admission assembly: the variable rename
unintentionally changed `item['record']` to a nonexistent dictionary key.
Both failures precede GPU/model updates and are preserved. Correct the exact
field access and check AST bindings/callable status writer; identical protocol
r3 is the first actual learning run. Launcher218720, training PID218779,
driver hash `0a223284fdb9e77c04d8b2e135091b47c4073a2815d37573053ef4a5899dc827`.
At step36 finite loss/gradient and video updates are confirmed; physical GPU0
3.76GB, reserved3.43GB, observed step samples about49--56% GPU utilization,
initial remaining ETA106s. This is a small-batch Probe, not a throughput
benchmark; do not add work merely to inflate utilization.

Initial frozen main panel: hand44.569mm/world20.269mm. Initial held video
point EPE21.028mm exactly equals the stationary baseline because the new point
head is zero-initialized; held actor130.457mm on209 valid hand points. These
refer to the single admitted video window, not the original five-session native
auxiliary panel. Source admission/clock/visual checks do not establish video
transfer. Actual r3 final metrics remain pending here.


## 17:35 final300 result and next cheapest Decision

r3 completed300 updates in104.097s, exit0. Video WORLD gradients reached
the shared backbone (mean norm0.24126); final parameter SHA
`fdc360eff0f50f05932edaebb01af7d4a1cd876fe5acdcc6f584dd277cdfca8c`. Latest/final share one
inode (629762368 bytes); all four remote GPUs are released.

Frozen held video scene EPE21.028→22.464mm (+6.83%); stationary21.028mm.
Held hand130.457→97.583mm (-25.20%,209 valid points in ONE window).
Main three-source hand macro44.569→46.648mm (+4.67%, <=5% screen PASS);
world macro20.269→21.847mm (+7.79%, <=10% screen PASS). True goals still
outperform static goals on all three sources. Scene >=5% improvement screen
FAIL; combined learning Probe remains UNCLEAR. This is not matched
causal evidence for video transfer or a negative method conclusion.

Local receipt: `tmp/video-training-20261012-takeover/mixed_probe_final.json`.
Remote artifacts: isolated scope/video-offline-mixed-20261012-r3/train/.
Measured isolated NAS5.133GB; last combined297.185GB, below proactive298GB
and Campaign300GB. Parent/native auxiliary final is preserved.

Decision: distinguish insufficient fitting from poor generalization before
longer training. Cheapest check: GPU evaluation of the fixed final on all
three train windows and one held window, with horizon/motion-bin EPE and
stationary baselines, without saving another checkpoint. Independently
extract/screen offsets0/3/6 from the other existing22 cached clips on CPU;
never rerun Omega or change the frozen session split/quality thresholds.
At most66 total extracted windows, <=200MB new arrays, <=20min processing,
GPU audit <=5min. Stop on source drift, foreign GPU process, invalid tensors
or storage threshold. Use admission counts and fit/static comparisons to
decide between more diverse data and a point-head/label diagnosis. No new
external identity, dataset download, branch or permission boundary is needed.
User requests continued work until19:30 local, with long-run polling <=20min;
interpret the typo “完善七点半” as evening19:30 on2026-10-12.
