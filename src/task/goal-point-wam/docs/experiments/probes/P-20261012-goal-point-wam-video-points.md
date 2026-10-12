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
the shared backbone (maximum post-clipping norm0.24126); final parameter SHA
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


17:43 GPU fixed-checkpoint audit: train windows EPE11.237/13.540/15.297mm
vs static10.957/14.290/16.457mm. Predictions move low-motion points about6mm;
<2mm-motion EPE5.43/5.36/6.18mm vs static1.43/1.40/1.36mm. Held similarly
6.01 vs1.37mm. Large-motion labels remain poorly fitted: train>50mm strata
98--117mm EPE vs108--128mm static. Motion bins are teacher-label audit strata,
not reliable physical motion classes. Do not label this purely generalization.

Diverse extraction r1 hit a stale5.1GB scope self-limit before any clip work.
Preserve its FAILED receipt; align r2 explicitly to5.4GB (<=6GB current scope
allowance, <=200MB additional extraction, global298GB active guard unchanged).

Decision: frozen-backbone point-head-only diagnostic on the SAME three training
windows, current final300 initialization,300 updates/lr1e-3, seed287. Cache
observed scene and action embeddings on GPU; no future labels in the cache
features. Optimize the existing masked point loss only. This distinguishes
insufficient point-head optimization from limited fitting under fixed
representation. Positive fit screen: train mean-window EPE >=20% lower while
masked loss decreases; held is reported but never used for selection. Save
only a small point-head diagnostic state, explicitly ineligible as full WAM
initialization; all original checkpoints untouched. Single GPU0 <=180s,
new artifacts<=10MB. Negative/inconsistent outcome means UNCLEAR and inspect
loss/decoder reachability; positive means pursue diverse admitted data rather
than blindly extending the tiny mixed fit. This is an exploratory diagnostic,
not a method comparison or claim upgrade.


17:46 frozen-feature head diagnostic completed300 updates in9.822s,
120323 trainable parameters; peak reserved367MB. Mean-window train EPE
13.358→8.217mm (-38.48%), masked loss also decreased: predefined >=20%
fitting screen PASS. Held22.464→25.715mm (+14.47%); no checkpoint selection.
Local receipt `tmp/video-training-20261012-takeover/fit_head_audits.json`;
remote diagnostic head is explicitly ineligible for full WAM initialization.
This supports point-head fitting under current frozen representation, while
showing stronger specialization to the one training clip. Next decision:
use a more diverse quality-screened cohort, not repeat the one-clip head fit.

17:48 extraction r2 deliberately stopped via its owned SIGUSR1 handler after
three completed windows; original arrays/receipts preserved. Its generic
20-second sleep unnecessarily delayed every small CPU child, risking the
20-minute bound before reaching held sessions. r3 uses proc.wait(timeout=20),
which returns immediately on child completion, and prioritizes frozen held
records. It reuses completed r2 manifests/audits and runs the same extractor
and thresholds for remaining clips. This changes supervisor waiting only,
not labels, split or scientific method; core remains9c81eaa2.


## Diverse cached-cohort mixed fit protocol (predeclared before launch)

Class Decision: test whether1250 video updates across multiple admitted
recordings improve held scene prediction while preserving the native main
representation. This serves reusable point-transition pretraining upstream
of Cm; it is not robot/RL validation. Small head-only fitting succeeded but
one-clip held prediction worsened, so adding diverse existing data is more
informative than repeating or extending that one-clip diagnostic.

Run_id video-diverse-mixed-20261012-r1; core commit9c81eaa2; model-only
initialization from original native auxiliary final10000, fresh optimizer.
Use ALL66 screened cached windows (22 source clips) from original and diverse
extraction; admit only original seven-gate PASS windows. Require>=6 unique
train sessions and>=3 frozen held sessions before launch. Frozen seed283
splits unchanged, no quality threshold relaxation or input selection using
future motion. Excluded windows stay in the manifest/receipts.

Single GPU0, main batch16/video4, 3native:1video,5000 total updates, lr1e-4,
model287/draw288, max2400s training/2520s supervisor. Frozen native64/source
panel, video validation every500; final fixed by step budget or safety stop,
never choose the best held checkpoint. Checkpoint interval10000 (>5000),
so save once at final and hardlink latest/final; avoid redundant terminal
checkpoint writes and an unnecessary second atomic-file storage peak.
Scientific retention screens: held video pooled EPE>=5% better than initial
and stationary; main hand<=105% initial, world<=110% initial. All PASS gives
a local PROMISING screen and motivates deeper per-session/motion analysis;
otherwise UNCLEAR and prioritize input/label/head diagnosis. Report per-session
values later, not pooled error alone.

Supervisor stops on main hand>110% or world>120% initial at intermediate
validation, nonfinite/OOM/input drift/foreign GPU/deadline/storage. These are
safety limits, distinct from the stricter final positive screens. GPU/ETA
observed on launch and subsequent polls; user-requested poll interval<=20min.
Existing local outputs+tmp292.053GB; isolated NAS~5.16GB. New checkpoint
<=0.70GB plus small logs/data projects combined<298GB proactive limit.
Remote supervisor cap5.85GB (<current self-allocation6GB), leaving~97MB
local headroom at that worst-case cap; recheck combined before launch and
during training. Campaign300GB/four-card maximum unchanged, use just one
remote GPU and preserve all original checkpoints.

Repository native scene features include observed point velocity/acceleration/
displacement; current video frontend zeroes them because it originally used
current-only geometry. Later cached windows may support observed track
history, worth a cheap separate check. This is OUR possible input adaptation:
PointWAM paper §3.1--3.2 describes current colored point cloud/current hand
keypoints, not a requirement to supply four-frame scene motion. Primary
source https://arxiv.org/html/2610.02840v1#S3.SS1. Do not claim that absence
of scene history invalidates the authors' current-input route or that ours
is a full reproduction. Keep this fit protocol fixed while investigating.

Admission finalized:66 total screened windows;54 train windows from18 unique
recordings and6 held windows from4 recordings pass. All54 train windows
pass;6/12 held candidates fail (4 new,2 original), preserved/excluded.
This selected held cohort is not the entire EgoDex distribution. Diverse
extraction completed60 windows in453.315s, CPU only, no reconstruction.
Normalized session names (optional .mov suffix removed) have no train/dev
overlap. NAS5.165GB before mixed fit, projected final combined297.85GB.


While diverse mixed fit runs, cheapest independent history Decision: CPU-only
reference forecast from teacher frames t-3/t for offsets3/6, on unchanged
current-selected points and frozen splits. Compare constant-velocity versus
stationary on the SAME past-known/future-valid points, first8 ticks and all24.
No extra model or checkpoint. At least3 held sessions, mean history coverage
>=50%, and held short EPE>=5% better than stationary motivates a paired
frozen-feature point-head history Probe. Otherwise defer scene-history
changes and inspect other signals first. Current-past alignment p90<=.25mm
and hashes must pass; bound CPU300s/JSON<=1MB. Interpolated past labels
are10Hz approximations, not native measured30Hz scene motion.


## 18:22 diverse final5000: UNCLEAR, no blind continuation

Run video-diverse-mixed-20261012-r1 completed5000 updates (1250 video),
1613.516s training/validation/save, exit0. Launcher268570/child268632,
fixed core9c81eaa2; driver SHA
`968a5181efea91df98b7c9ce34b5ef75ab6dede446e847e52a64c4d53973dde8`.
Supervisor checks ownership/input/foreign GPU/storage/intermediate native
degradation about every20s; no stop condition triggered. Peak physical
GPU4659MiB; all remote GPUs released at completion.

Held six-window scene EPE14.776→16.086mm (+8.864%, static14.776): FAIL
>=5% improvement criterion. Held hand77.088→77.636mm (+0.711%).
Native macro hand44.586→46.042mm (+3.265%, <=5% PASS), world20.290→
20.658mm (+1.811%, <=10% PASS). True goal advantage retained on all3 sources.
Fixed final parameter SHA
`3e5125c95bbeddd2829b2c0ea4ce39d2a339af5654602b47539937b8eba862d9`.
Latest/final are the same inode,629774336 bytes. Final validation is repeated
at loop boundary and exit; tiny floating differences in world/actor metrics
are retained in both lines, use final line consistently. No held-selected
checkpoint or comparison to the one-clip panel as though it were matched.

Local receipt `tmp/video-training-20261012-takeover/diverse_mixed_final.json`;
remote scope/video-diverse-mixed-20261012-r1/train/. NAS5.804GB; local
outputs+tmp292.067GB, combined~297.871GB. No new full checkpoint allocation
without reclaiming eligible own disposable material or rechecking headroom.
All scientific checkpoints/arrays preserved.

Next: fixed final train/held per-window/motion-bin audit (GPU<=300s, JSON only),
then decide whether remaining issue is fitting, action use or geometry/noise.
CPU history-reference r1 failed before computation because graspenv lacks
h5py imported by geometry utilities. No environment installation or method
change: r2 uses the existing preprocessing environment already containing
h5py. Preserve failed log; this is an execution/environment defect, not
negative evidence for point history.


18:30 cheapest action-use Decision audit: fixed diverse final5000, same scene/
H4/validity; replace future actions with current hand repeated24 times and
zero current-relative/step deltas. Report world response, actual-label EPE
under true/static actions per-window/session. Actor prediction must remain
unchanged within1e-4m (numerical tolerance); larger change is an implementation
blocker because future executed actions belong only to the world branch.
If actual actions do not help on held recordings, diagnose weak action use/
generalization before more representation training. No claim that output
response alone demonstrates physically correct actions or Cm benefit.
GPU0 <=300s, no weight updates/checkpoint, JSON<=1MB.


## 18:37 fixed-input diagnostics and early-window implementation fix

Final5000 window-macro train scene7.347mm vs static8.055mm (-8.80%);
held window macro16.346 vs15.074mm (+8.44%). Four held session macros:
basic-pick-place23.518/static21.028; bench-chair13.145/11.004; bench-lamp
14.786/14.251; cloth20.336/22.151. Only cloth improves (~8.19%); do not
claim all domains ineffective or hide the three negative sessions behind
a pooled metric. Train hand window macro9.620mm, held82.740mm. These
window/session macros differ from training evaluator point-pooled values.

True/static action intervention: train world7.347/7.740mm; held16.346/
15.803mm. Mean response1.579/3.737mm, proving action dependence, not
physical correctness. Actor max change train3.052e-5m/held0 due to repeated
sparse floating computations, below1e-4 tolerance; no future-action leak
found in inspected branch. Exact wiring evidence comes from source and
input intervention together, not gradient/response alone.

Past-track reference screen FAIL: held mean short8-tick CV19.314mm vs
static11.074mm; full24 CV43.011 vs15.969mm. History coverage99.61%, current
alignmentp90 exactly0. Thus no history feature expansion now; this does not
prove a learned history encoder cannot help. Geometry/noise and actual
acceleration confound this simple extrapolation.

Coverage bug: prepare_video_points previously required clip_start>=3 even
when current=clip_start+offset had valid nativeH4. Fix clock gate to current
tick; source f0 offset0 still rejected, offsets3/6 may enter. New pure
video_clock helper validates11 integer10Hz frames, nonnegative source,
allowed offsets, currentH4 and completeK24 coverage; same existing nonzero
clock preserved.16 relevant tests pass (7.52s), including actual boundary
contracts, invalid clocks and prior point-loss/input-scope tests. This is a
coverage fix, not relaxed geometry quality or future-frame input leakage.

Decision: CPU-only extract/screen previously omitted cached teachers mapped
back to original24 native records/frozen seed283; prioritize initial clips
and other existing cached intervals. Same seven gates, max60 new windows,
<=35MB actual scope growth, <=15min CPU; scope cap5.84GB/combined298GB
stop unchanged. Separate immutable new plan/run, no completed scientific
manifest or checkpoint mutation. This tests whether avoidable coverage loss
excluded more informative motion/held sessions before further model work.


18:49 early extraction setup: isolated clone initially lacked fetched-only
commit45632 in its reachable refs and unnecessarily copied83.18MB objects.
No extraction had started. Configure this new clone's object alternates to
the existing owned video-repo object store, verify9c81/45632 commits, remove
ONLY its redundant newly created clone pack/index/reverse-index, and checkout
45632. Original code tree, objects and checkpoints untouched. Narrow sparse
paths to actual audit/common tools. Scope returned to5.807GB.
Launcher281770/driver SHA
`21adc2af081762af106f77b4bdc6d71b6f9a80cb873634c7d3ec38c6b603e5da`.
New mapped cohort23 cached teachers/50 windows/23 records (18train,5held
teachers); four other teacher archives cannot map to frozen original raw
records and are excluded. Mapping reads existing pack source_episode_id; no
filename-based guessed identities or new download. Plan on scope/video-early-plan.json.

Decision: current-depth normals as compatible native input fields. Native
18D point features reserve dimensions0:3 for measured surface normals;
video currently zeros them. CPU surface-normal estimate uses ONLY current
metric depth/intrinsic and current tracked UV, camera-facing orientation,
finite/support/discontinuity masks, unit norm; unknown stays zero. Analytic
fronto-parallel/slanted planes and NaN support checks before model probe.
Two paired frozen-representation head fits: original zero normals vs current
depth normals; same final5000 parent,54train/6held admitted windows, identical
2000 cyclic/random draws (draw288), head-only AdamW lr1e-3. Cache both
feature arms on GPU0. Native backbone/actor/goal/action/rigid world unchanged;
no full WAM checkpoint. Positive screen: held point-pooled EPE>=5% better
than zero-normal arm and stationary, and >=3/4 held session macros improve.
Else UNCLEAR, defer normal field integration. Fixed finals, no held-based
selection. GPU<=300s and artifacts<=2MB (small typed diagnostic heads/JSON);
stop on drift, foreign GPU, nonfinite/deadline/scope/global budgets. This is
a single-seed input/optimization Probe, not official PointWAM reproduction
or formal Cm/robot evidence. Current PointWAM paper input includes colored
geometry, but this normal field specifically adapts OUR native feature contract.

## 19:10 early coverage and paired-normal outcomes

Early extraction completed50 windows in325.233s;45 passed unchanged gates
(40train/5held). Combined existing/new cached coverage is105 qualified windows:
94train/19 sessions,11held/4 sessions. The fifth held session still fails
quality; no threshold relaxation. Independent raw-HDF clock audit checked all50
windows, including quality failures: native H4/K24 hand mismatch0m; directly
backprojected next10Hz point mismatch max3.257339e-7m. This supports source
clock/coordinate implementation, not teacher physical accuracy. New individual
engineering manifests remain training_allowed=false; completed scientific
manifest54train/6held is unchanged. Receipts: early_normals_final.json and
early_clock_receipt.txt under local tmp/video-training-20261012-takeover/.

Normal paired diagnostic completed60.957s, peak reserved574.620MB. Held pooled
zero-normal14.88001mm vs current-normal14.82282mm; stationary14.77599mm.
Only2/4 session macros improve; predeclared5%/stationary/3-session screen FAIL.
UNCLEAR: defer product normal integration. This comparison does not attribute
head-only optimization gains over the full-model final to normals. Typed small
head files cannot initialize the full WAM. Parent full-checkpoint SHA
573701835d1ac8b2c48d01efd761893e2cbcb2aa98164f129796ef174fb23126.

Decision: paired motion-weight diagnostic, same frozen final5000 parent,
54train/6held, current zero-normal features cached once, same initial head,
2000 draws seed288, AdamW lr1e-3, gradient clip10. Original arm uses cumulative
current-relative displacement/2mm; alternative uses adjacent-frame target
displacement/5mm, first predecessor=current query. A missing predecessor
gets zero motion weight while current valid supervision remains; NaNs are
masked before arithmetic. Hold Huber/.05m, .1 weight floor and per-time/window
reduction fixed. This tests a weighting adaptation, not full paper B.5 (which
also differs in floor/reduction/MSE/statistical scale/clipping); definition
and threshold change together. Independent primary-source note:
paper/pointwam/video-loss-contract-audit-20261012.md.

Question: does this weighting adaptation reduce held false motion enough to
justify a later coupled-model Probe? Positive screen: alternative fixed-final
held pooled EPE>=5% better than original and stationary, >=3/4 held session
macros improve. Otherwise UNCLEAR and no blind full continuation. Log motion
strata as label-defined diagnostics, not physical motion truth. GPU0<=300s,
artifacts<=2MB, no full model/checkpoint mutation; stop on nonfinite/drift,
foreign GPU, deadline or storage envelope. Run video-motion-head-probe-20261012-r1.

19:14 Decision audit: same original RGB frame reconstructed in overlapping
cached teacher clips. Current extraction is coherent within one teacher pack,
not automatically throughout an entire multi-minute recording. Compare already
cached metric depth at identical pixel coordinates/shared raw frame IDs for
paired clips of the same raw episode; use existing scale/intrinsic only, no
refit/reconstruction/input modification. Pixel grid<=4096/frame, finite positive
depth masks, report per-frame/episode median/p90 3D disagreement and existing
per-clip scale. CPU<=300s, JSON<=1MB. Exact raw identity/clock/intrinsic required.
This asks whether clip-context reconstruction inconsistency remains large
enough to prioritize unified trajectory geometry before more model fitting.
The two reconstructions do not provide physical ground truth; disagreement
can locate a consistency problem, agreement cannot prove accuracy. Compare
to the existing 10mm-median/25mm-p90 quality envelope descriptively, without
retroactively changing admission or formal conclusions.

## 19:15 paired weighting outcome

Motion-head diagnostic completed72.843s, peak reserved505413632 bytes (~482MiB),
driver322315/child322345; driver SHA
d52bb420dc3acd9801952cc0c3915e360437af09b069745f34b12097d82fd20e.
Held pooled original14.880007mm vs adjacent14.845424mm (-0.2324%);
static14.775990mm. Only1/4 session macros improves; positive screen FAIL,
UNCLEAR. Held label-motion bins (<2,2–10,10–50,>=50mm) original/adjacent EPE:
2.2363/2.3446,5.6555/6.1536,18.6259/18.5969,149.9335/142.2750mm.
High-motion tail improves while low-motion errors worsen; no evidence this
weight adaptation resolves held false motion. These bins are reconstructed
label magnitude, not verified object motion. Same parent/backbone/head init
and draw order; original arm exactly reproduces the normal Probe zero arm
14.880007mm. Small typed head states only; no full WAM checkpoint or method
negative claim. Local motion_head_final.json; remote run/fit/result.json.

Cross-clip audit r1 failed before comparison because old/new plan records use
different clock-field schemas. Preserve empty failed output; r2 reads actual
teacher source_frame_ids (authoritative clock), unchanged comparison method.

19:18 cross-clip result:45 mapped teachers/19 overlapping episode pairs/114
shared raw frames;102 frames from17 pairs have both teacher clips represented
by quality-admitted windows. Identical grid pixels/raw camera pose checked.
Among both-admitted frames, per-frame median disagreement median12.146mm,
p90 disagreement median26.687mm;61/102 exceed10mm median,54/102 exceed25mm
p90. Episode medians range~2.06 to56.16mm. The numerical alignment fix within
each cached teacher is valid, but whole-recording geometry is not yet unified.
This is direct reconstruction inconsistency, not ground-truth reconstruction
error or established causal attribution for poor training. Receipts:
cross_clip_geometry_final.json; remote video-cross-clip-geometry-audit-20261012-r2.

Decision: prioritize teacher-context consistency over another feature/loss
training sweep. Cheapest follow-up separates multiplicative scale from depth
shape: per clip pair fit ONE median depth ratio on alternating fixed grid
pixels across all shared frames, evaluate remaining grid pixels; additionally
evaluate per-frame scalar as a descriptive upper flexibility reference. Preserve
uncorrected metric depth/report, no teacher or scientific label mutation.
If one scalar leaves most mismatch, unified scalar calibration alone is
insufficient; investigate one geometry per raw frame/full-record context before
coupled training. CPU<=300s, JSON<=1MB. Shared-frame pixel split is a numerical
consistency audit, not a held-recording scientific experiment or physical scale
validation; held labels never select a scientific checkpoint.

19:20 scale-vs-shape result: on alternating evaluation grid pixels, both-
admitted shared-frame median-of-medians12.207mm original→3.967mm with ONE
pair scalar (~67.5% lower); p90 median26.637→9.934mm. Per-frame scalars only
slightly improve3.877/9.694mm. Scale inconsistency is therefore a substantial
numerical component, while chair/lamp pair medians remain11.636/10.873mm.
Agreement optimization is not physical scale validation. No corrected labels
admitted and no retroactive reinterpretation of the previous training.

Engineering next: test a single reconstruction context over the complete
cached span of ONE train recording basic_pick_place/0, source frames0:3:45
(16frames), using pinned Omega/direct weights and original camera/intrinsic.
Calibrate ONCE over that span using existing original hand labels; store one
depth per raw frame and immutable context identity. This is a feasibility smoke
for the user's trajectory-first workflow, not a complete multi-minute recording
or fresh scientific fit. Successful finite output and recoverable calibration
permit designing the whole-recording shared geometry contract; poor calibration
or OOM stops the approach without scaling. No tracker rerun/full checkpoint.
GPU0<=360s, new artifacts<=20MB, scope cap5.855GB; current combined297.933GB
leaves ~67MB below proactive298GB, and global300GB hard bound unchanged.

## 19:27 trajectory-first feasibility and extraction outcome

Single-context smoke completed54.676s, peak allocated7666918912 bytes
(~7.14GiB); driver323549/child323579. All16 native source frames0:3:45 have
one finite positive metric depth. One scale=.597694, relativeIQR=.349624
(just below unchanged.35 gate; not robust physical scale evidence),320/320
calibration inliers. Depth archive SHA
59f42b78f4a8923abf2c899cb9548afab1d1f18925d1ef2aeea8f2b4c830056e.
This spans1.5s of ONE train recording, not its entire multi-minute trajectory.

Implemented --trajectory-depth with shared_trajectory_depth: verify declared
engineering/offline scope, original video identity, exact raw frame coverage,
depth/metadata hashes, camera/intrinsic and finite positive depth. Use that
fixed depth for BOTH current input and future labels; no depth inference,
window scale refit, offset subtraction or teacher mutation. Full shared span
context/calibration is preserved in window provenance. Audit uses the same
shared geometry and retains comparison against the original clip reconstruction.
Four new contract tests cover bit-identical depth for overlapping windows,
missing-frame rejection, source/camera mismatch and depth hash drift;
18 relevant shared-geometry/clock/point tests passed8.02s. Existing default
clip route is retained for historical identity/reproduction.

Remote isolated cached-trajectory-repo pinnedb317651c4bbb6fa8738771ab67ebaca85cf297d8
(owned helper7fbb147a, source/tests/notes included by concurrent research
snapshotea55c66e, READMEb317651c). Historical9c81/45632 trees and artifacts
untouched. video-shared-geometry-windows-20261012-r1 driver323943 extracted
two real windows from old f0/f15 tracking clips with the shared depth,12.380s
CPU, no model/calibration rerun. BOTH pass original seven quality gates;
current input/label disagreement exactly0mm. Background candidate maximum
median/p90: first5.130/12.589mm, second7.032/16.343mm. They remain engineering
manifests; no scientific fit or causal-online/physical-accuracy claim.

Next decision: this validates trajectory-first geometry reuse over a bounded
span. Extend consistent reconstruction/calibration and identity to longer
recordings before another admitted matched video fit; limit GPU/memory/storage
explicitly. Full-record Omega context and chunk boundaries remain unresolved,
as do approximate hand-to-surface scale and teacher physical accuracy. Do not
declare the previous poor scene fit solely caused by this discrepancy.

Local receipts shared_geometry_final.json and cross_clip_scale_final.json
under tmp/video-training-20261012-takeover/. Compact comparison visualization:
outputs/goal-point-wam/video-diagnostic-summary-20261012-r1/diagnostics.png
(head arms paired; full5000 uses a different optimization recipe).
Remote scope5.854GB, local outputs+tmp~292.097GB, combined~297.951GB;
no further full-checkpoint allocation. All four assigned remote GPUs released.

19:29 final engineering check: pass one new shared-geometry window to the
existing pinned9c81 mixed trainer --smoke (4updates, video actor masked to
isolate world gradient, no checkpoint). GPU0<=120s, log/JSON<=200KB; existing
native data/statistics and auxiliary parent unchanged. This only verifies
the new data reaches the shared backbone with finite gradients, not held
prediction quality or scientific video benefit.

19:31 closeout: r1 smoke used unsupported --main-data and failed at argument
parsing before a model/update. Preserve failed run; r2 corrected --data,
completed4 mixed updates in3.910s (full launcher ~22.6s), no checkpoint.
Driver325887/child325917, all losses/gradients finite. First video update has
zero backbone gradient because the new point output layer is zero-initialized;
second video update reaches shared backbone norm1.66559635e-5. World-only
video gradient isolation passes; this is engineering wiring evidence only.
Peak physical GPU2869MiB, GPU now released. Local receipt
shared_geometry_smoke_final.json. Final local outputs+tmp292.113GB and
remote5.854GB, combined~297.967GB. No active video job left unattended.
Continue-to19:30 user request fulfilled; no additional scientific conclusion
or successful generalization claimed. Next work is longer-record shared
geometry/calibration plus separate-session admission before a matched fit.

## 19:45 long-recording reconstruction feasibility

User explicitly permits expanding storage and requests an actual assessment
of long-recording trajectory reconstruction. Decision Note: temporary extra
10GB video-only artifact allowance (combined ceiling310GB; owned remote scope
ceiling15GB), no deletion of previous scientific data and no machine change.
This supersedes the previous300GB/298GB stop for this bounded video work only.
Actual remote NAS free~54.3TB; local filesystem free~652GB. Prior298GB was
an interpreted experiment-output budget, not physical disk capacity.

Raw metadata correction: current22 mapped records are2.07–40.17 seconds,
not assumed multi-minute recordings. All raw video/native-camera counts agree
at30Hz. Longest cloth held record:1205raw frames/402sampled10Hz frames;
longest train cloth record860raw/287sampled. Earlier shared span only0:3:45.
Pinned primary Omega source48b23/weightsc02da; published reference memory
400frames35.71GB and500frames43.15GB at512-balanced, different hardware;
our idle GPU exposes49140MiB. Source global attention still couples all
frames; no recurrent streaming state in this forward API. References:
https://github.com/facebookresearch/vggt-omega#runtime-and-gpu-memory and
https://arxiv.org/abs/2605.15195 . Estimates are not measured local feasibility.

Engineering/Decision Probe: reconstruct the entire longest acquired recording
ONCE at the existing10Hz scene clock/512-balanced, all402 frames and original
native camera/intrinsic. No independent window reconstruction/calibration.
Use actual native2x11 hand positions with confidence>=.5 for one global scale;
record this anchor choice explicitly (earlier cached teachers used derived
2x10 palm/tip points). Report measured startup/forward/save time, peak GPU,
global scale support/IQR and temporal segment scale statistics. Fixed offline
context, no model training/checkpoint selection or scientific val metric.
GPU0<=20min, artifact<=2GB, owned scope<=15GB; stop on OOM, nonfinite,
input/code/vendor drift, foreign GPU or timeout. A failure distinguishes
compute/memory feasibility from quality. Successful completion alone does
not establish reliable long point tracks or ground-truth metric accuracy.

19:53 full40.17s depth run completed:402frames/384x688 model inputs,
forward99.759s, preprocessing+model+save123.681s; launcher wall138.978s.
Peak allocated39.071GB (~36.39GiB), no OOM. Single native2x11 calibration
scale=.395173,5359samples/5115inliers, relativeIQR=.175119 (<.35 PASS).
Ten time-segment median ratios relative to global .9602–1.0286; no monotone
drift in this numerical hand-anchor audit. This cannot exclude joint/surface
bias or certify true metric depth. Whole-record geometry SHA
f523fa80eb4633c901cf5cd61248362fb985483f541775cae8d952a9709d5917.
Run video-long-depth-20261012-r1; launcher327183/child327231; source3d861ed3.
This shows whole-record single-context geometry is computationally feasible
for this acquired longest recording at10Hz; no claim about minutes/30Hz.

Next engineering/Decision: full402-frame CoTracker3 offline on the same RGB,
fixed1024 queries:512 evenly spaced current pixels born atframe0 and same
512 born atsampled-frame200 (20s). Mask all pre-birth outputs. Adding later
queries tests whether loss of initial-point coverage can be addressed without
reconstructing depth again; point births use fixed current grids, never future
motion/visibility selection. Model forward keeps entire temporal context;
only independent convolutional feature extraction uses supported chunk_size32
to reduce temporary memory (no model/vendor edits or temporal resets).
Pinned source82e02e8029753ad4ef13cf06be7f4fc5facdda4d, offline weights SHA
2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834.
GPU0<=10min, artifacts<=100MB; reuse already saved depth/camera/one scale.
Report visibility/valid3D per birth group over time and1s window candidate
drift, with no physical-static or point-ground-truth claim. CoTracker forcibly
sets query-birth pixels/visibility, so birth agreement cannot validate tracking.
Engineering output only; neither held-label admission nor training is automatic.

19:58 tracking r1 OOM in legacy explicit time attention (q@k.T), requested
5.41GiB with1.50GiB free; process45.86GiB total, allocated34.70GiB plus
10.72GiB reserved-unallocated. It reached all13 feature chunks; independent
feature chunking is insufficient for1024 simultaneous point queries. No
tracking accuracy result produced. Preserve r1 log, not method-negative evidence.
Next r2 uses four256-query batches, each still sees all402 frames with the SAME
fixed global depth/scale. This follows existing pilot query-batching semantics;
spatial attention support changes between batches, so exact output invariance
is not claimed. No external vendor patch/new weights/temporal window reset.
GPU0<=10min, output<=100MB. Further OOM would justify128 or efficient attention,
not rerunning independent depth/calibration per training window.

20:02 r2 completed:1024 queries over all402 frames in11.483s tracking forward,
18.844s including decode/load/lift/save; peak allocated17.773GB (~16.55GiB).
Four256-query batches solved the observed temporal-attention OOM without
new depth/scales or temporal resets. Sourcecf2c60c1, driver328862/child328893.
All pre-birth outputs masked and valid tracked XYZ finite. Receipt
long_video_tracks_final.json; remote video-long-tracks-20261012-r2/tracking/.

Final engineering reference: extract H4/K24 windows at raw currents3/303/603/
903/1179 from this SAME full-depth/full-track cache, selecting points only by
current validity/workspace/voxel. Original native2x11 hand clock/confidence;
10Hz point measurements interpolated30Hz, null object goals, offline context
declared. CPU<=60s, <=5MB window artifacts, engineering-only; verify bit-identical
current label/input positions and actual loader admission without model inference.
This tests whether trajectory-first work makes subsequent window extraction
cheap and consistent; it does not certify labels or change frozen scientific
cohort/checkpoint. Only sampled visible tracks enter each current observation.

20:07 whole-cache extraction r2 COMPLETED: five actual H4/K24 windows at
0.1/10.1/20.1/30.1/39.3s, selected current point counts503/155/518/365/396;
CPU1.530s including strict reference loader+collate for all5 windows. Current
input/label positions bit-identical, and independent UV/depth reprojection
agrees exactly (0mm in this numerical same-cache check). Native hand time
contract retained. No depth/tracker/calibration rerun. Engineering loader
acceptance does not mean quality-gate admission or scientific training.
Manifest SHA4f40fb52945a0a317b25e3ac40b9632d38c0c74e58db8ed3459a7e2d2753c7ef;
script SHA6017a38da31e2f24ede8f122a933aa15ae72aa5a91df49806ae1fcf94a12f511;
reference loader SHA7ceb9e17517bcec67cfa727322ebfb8236ca94241f21da0285a1627ea7039185.
Run video-long-window-extraction-20261012-r2. Extraction r1 produced window
files but package import unnecessarily pulled training spconv into prep env;
retain failed log/data. R2 loads the exact immutable video_data.py module via
importlib without importing unrelated model initialization; no mock dependency,
loader implementation change, environment install or window math change.

Decision result: trajectory-first geometry and point tracking are feasible
for the acquired40.17s recording at10Hz on the assigned49GB GPU. Initial1024
simultaneous query tracking OOM is resolved by256-point batches. Whole-record
context/calibration removes separate-window numerical frame disagreement by
construction; this is not evidence of physical depth/track accuracy. Full30Hz
1205-frame or multi-minute single-pass reconstruction remains untested.

Remaining blocker to broad training-window admission: occlusion/coverage,
not extraction or disk capacity. At10s only107 initial points valid3D (10
low-image-residual background candidates); 10.1s extracted155 points, below
existing256 minimum. At35s initial75 plus161 later-born points=236. At25s
later-born points contribute275 beyond initial183, supporting periodic point
births within shared geometry as next cheapest step. Birth forcing and later
reappearance preclude interpreting birth agreement or sampled count as track
accuracy/monotonic survival. Some windows lack enough candidates for the
original background gate; no gate relaxation or automatic training admission.
Next worthwhile engineering Probe: denser time-spaced fixed-grid query births
on this same geometry cache, then existing quality gate; no new model fitting
until admitted separate-session labels exist. Metric ground truth and actual
held-session training benefit remain future evidence, not inferred here.

Local numeric summary and inspected plot:
outputs/goal-point-wam/video-long-record-summary-20261012-r1/summary.json and
coverage-and-scale.png. Local receipts remain under
tmp/video-training-20261012-takeover/. GPU depth/tracking jobs completed and
released assigned GPU; no video model-training/checkpoint created in this spike.

## 20:22 short-window query restart with shared whole-record geometry

User approved the proposed hybrid: immutable whole-record depth/calibration,
then fresh queries in short windows, and also requests MINT/HaWoR source review.
Bounded Decision Probe: discriminate accumulated long-track point loss from
intrinsic inability to label local1s motion. Eight raw starts0/150/300/450/600/
900/1050/1170, 11frames at10Hz; native current=start+3 and retained H4/K24.
Compare current-selected point counts with the previous full-record tracker
at these identical ticks and run unchanged audit_video_tracks.py quality gates.
If coverage and quality pass, this becomes the next label-extraction workflow;
if only coverage improves, investigate geometry/label quality rather than
expanding training or weakening thresholds. Frozen held recording and official
TRAIN acquisition provenance retained; no held-session fitting or cohort update.

Cheapest adaptation: every-second-pixel candidate grid, select up to2048
identities by first-frame depth and1.5cm voxel ONLY before model tracking;
256-query batches, each sees the complete11frame clip. This is not exact
PointWAM: paper subsamples after tracking, retains up to12000, and restarts
clips every0.5s. Here8 predetermined starts and pre-subsampling bound compute;
no selection by future visibility/motion, no depth or scale recomputation.
Each clip has local track IDs; no cross-clip identity/global motion claim.
Teacher archives include bounded shared-depth slices for compatibility with
unchanged auditor; these are cached values, not new depth predictions.
Engineering loader acceptance remains distinct from label admission/training.

Run video-window-restarts-20261012-r1: one assigned GPU0, <=10min model+data
budget/<=15min supervisor deadline, <=250MB owned artifacts, owned remote
scope<=15GB and previously user-authorized combined ceiling310GB. Stop on
input/source/vendor drift, OOM/nonfinite, foreign GPU, resource cap or deadline;
retain partial output/logs. CPU quality audit and loader are file/tensor work,
not CPU model inference. No checkpoint/model-training budget in this Probe.
Implementation tools/audit/probe_window_video_tracks.py; exact commit pinned
before execution. Additional MINT/HaWoR primary-source research runs independently
without weights/GPU, and cannot change this already-fixed run protocol.

R1 failed after completing the first2048-query clip: progress logging reused
`seconds` both for elapsed time and recording timestamp, raising TypeError.
Peak allocated1.151GB, first-window teacher/sample retained. No method result
or quality admission from incomplete run. Fix uses elapsed_seconds separately;
r2 repeats the fixed8-window protocol in a new output and pinned detached repo.

20:28 r2 COMPLETED, source023de78f:8/8 unchanged seven-gate admission screens
PASS; strict engineering loader/collate all8 PASS. Tracking5.126s, total16.420s,
peak allocated1.151GB. Baseline vs short fresh selected counts at identical
native currents0.1/5.1/10.1/15.1/20.1/30.1/35.1/39.1s:
503->1863,256->1804,155->1346,193->1743,518->1332,365->1642,243->993,
401->1829. Future valid fractions40.56–91.43%; numerical same-frame alignment
0mm for all. Background candidates368/717/39/1086/184/674/42/544;
all candidate median<=10mm and p90<=25mm. The10.1/35.1s background support
is only39/42, close to32 gate; still not physical-static proof. Queries were
increased from512 per sparse birth group to2048 per local clip and temporal
context changed, so these results test a practical composite workflow, not
an isolated causal effect of track lifetime. Geometry/calibration hash unchanged.
Manifest SHA5dfe520efacc70950c57cdc1314e7a111888f980b3f3487c536fe2db70e7f15c;
run72.456MB, owned remote scope6.415GB, assigned GPU released. Label-engineering
route PROMISING for these8 windows; no training/physical accuracy conclusion.

Next smallest Decision Probe: apply the fixed recipe across this same acquired
longest recording, rather than selecting only good windows. All1s clips with
0.5s stride plus final shifted tail (80windows, finalrawstart1173/sourceend1203).
Keep H4/K24,2048 initial-only query cap,256 batches, shareddepth hash and all
original quality gates. Report full admission fraction and weak temporal spans;
>=80% screens passing supports rollout to a distinct train recording, otherwise
inspect failing gates before expansion. Complete tail/globalIDs borrows MINT's
window organization, not its predicted camera stitching. GPU0<=10min model/data,
supervisor<=15min, new output<=1GB, scope<=15GB/combined<=310GB. No data deletion,
no hand/camera reconstruction and no training; preserve all failures. Primary
MINT/HaWoR audit saved paper/pointwam/mint-hawor-pipeline-primary-audit-20261012.md;
root independently checked camera chaining and imputed-valid source lines.
