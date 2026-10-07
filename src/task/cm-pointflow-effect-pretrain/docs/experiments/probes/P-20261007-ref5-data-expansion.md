---
schema: ref2dex.probe.v2
probe_id: P-20261007-ref5-data-expansion
experiment_id: P-20261007-ref5-data-expansion
date: 2026-10-07
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 616d4132a25b86619939e067e3595c0670e5940d
claim_id: C3
hypothesis_family: HF-ref5-data-expansion
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: expand native 3D sources first and retain EgoDex object reconstruction for a bounded pilot
decision_changed_if_negative: resolve the failing source contract before corpus expansion
status: UNCLEAR
run_id: ref5-data-expansion-20261007
---

# Ref5 native-3D expansion and EgoDex engineering pilot

Decision: test whether existing GRAB/ARCTIC and native EgoDex hands can feed
the current 30 Hz / 24-future-step hand/object contract without changing the
live models or training data. This serves the Task's multi-source effect
pretraining subgoal; it does not test Cm policy utility. The cheapest check is
bounded conversion plus pose/geometry/time/mask auditing, before any fitting.
Classification: Blocker engineering preflight, not scientific validation.

User explicitly requested [ref5](../../user/ref/ref5.md). Keep the current
branch and other agent's sources, configurations, training, checkpoints and
shared documents untouched. Own only `tools/run/ref5_data_expansion/`, the
dedicated tests, this card, `docs/REF5_DATA_EXPANSION.md`, and the output root
`outputs/cm-pointflow-effect-pretrain/ref5-data-expansion-20261007/`.
Base commit above plus actual script SHA256 identify uncommitted execution;
do not make a shared-index commit while another agent is modifying this branch.

## Question, decision and limits

Hypothesis: native 3D labels admit metric, stationary-frame 0.8 s windows;
ARCTIC requires separate rigid articulated parts. A passing native conversion
justifies this lower-cost source expansion. EgoDex passes only after object
6DoF, geometry and camera/world alignment are actually verified; native hands
alone do not constitute effect data. Never replace unknown object labels with
identity poses or reuse human hand reconstruction on EgoDex.

Resources: <=20 GRAB and <=20 ARCTIC sequences; <=20 native EgoDex test clips;
<=2 GiB selected EgoDex files/transfer; <=12 GiB all new data, dependencies and
models; >=20 GiB free disk reserve; acquisition<=1800 s and native conversion
<=1200 s. At most one additional GPU (GPU3 after checking that the other agent
has released it), with the live GPU0/1/2 training counted against the global
four-GPU cap. Pure file/label/statistics work uses CPU. No predictor/PPO fitting.
Source data outside this repo remain read-only. Stop on input drift, nonfinite
labels, ownership conflicts, budget exhaustion or unavailable required assets.

Domestic mirror is probed before official-download proxy fallback. Use official
ZIP byte ranges to select pairs rather than materializing 17.3 GB, preserving
member paths and CRC32 plus local SHA256. The archive size/ETag and full member
inventory are recorded. EgoDex official test is ENGINEERING ONLY; it must never
be appended to training or used for checkpoint selection.

## Deferred evidence

Larger source-balanced splits, duplicate-source-episode audit, more objects,
matched training, robot transfer and multi-seed validation follow only after
the engineering contracts pass. No generalization or utility claim follows
from these conversions. Estimated EgoDex object trajectories remain distinct
from mocap ground truth and require quality flags and scale/alignment checks.

## Authorized object deployment extension

User selected "continue deploying the free ObjectForesight pipeline" during
this session. Deploy into `.venv-object` and `vendor/` under the same output
root; existing environments and external projects stay read-only. Set an
additional <=24 GiB setup/model/output cap, >=20 GiB free disk reserve,
<=2 h initial deployment/download and <=90 min GPU3 inference cap, within
the global 300 GB/four-GPU campaign. Check resources before each model stage.
Start with one rigid tape-measure clip, then expand only after seeing actual
mask/mesh/depth/pose outputs. No paid API, Docker/system modification, human
hand re-estimation or predictor fitting. Manual object clicks replace automatic
EgoHOS/VLM selection for this engineering pilot, preserving the free
SAM2 -> TRELLIS -> SpaTrackerV2 -> FoundationPose object route.

## Native conversion corrections and audit

`native3d/` failed at GRAB MANO batch shape: explicitly expand zero default
betas alongside the subject template. `native3d-r2/` has ARCTIC mesh values
left in mm and zero accepted ARCTIC windows; it is invalid for training.
`native3d-r3/` and `-r4/` investigate a deeper part-label mismatch: official
ARCTIC `common/object_tensors.py` adds one to raw `parts.json` and articulates
`parts_ids==1`, so raw0 is top, raw1 bottom, top rotates about negative z.
The older external local adapter reverses these labels and is not an
authoritative geometry reference. New code corrects this without changing it.
The final `native3d-r5/` retains exact triangle indices and barycentric
coordinates, allowing all512 sampled points/part to be checked against their
original part geometry. This avoids numerical instability in nearest-point
projection on the extremely thin ARCTIC ketchup triangles; earlier proximity
audit failures are retained. No rejected output is silently overwritten.

Final native engineering audit:20 GRAB +20 ARCTIC sequences;3650 +23105
overlapping anchor windows, train17998/val3667/test5090. Subject splits are
disjoint per source. Official GRAB ObjectModel and independent official ARCTIC
quaternion operations agree with all40 sampled sequence pose/geometry checks;
WM rigid correspondence maximum error7.30e-8m, across deterministic sampled
windows in all splits. These are engineering checks, not statistical evidence
that more data improves prediction or policy performance. No program labels
are fabricated. Original frame IDs/timestamps are preserved.

## EgoDex native labels and isolated deployment

20 candidate pairs (40 members) were acquired with 63,059,073 transferred
bytes, preserving ZIP CRC and local SHA256; `egodex/download_manifest.json`
contains source/mirror/proxy evidence. Native conversion preserves 2,834 frames
and 2,294 hand-only windows. No object effect is fabricated, and the official
test split remains `training_allowed=False`. Metadata curation currently
identifies8 rigid candidates,7 requiring visual/material review and5 excluded
plush/articulated targets;20 acquired clips do not mean20 verified rigid tracks.

Tape-measure `basic_pick_place/101` is the first target. The initial small
patch SAM2 prompt is rejected; the whole-object box in `masks-r3/` covers the
inspected tape measure across231 native-rate frames, with occlusion quality
still needing downstream checking. The red display is mask overlay color.

Deployment stays under the owned output root, with read-only inheritance of
the existing `robo3d` environment. Runtime packages are pinned in
`deployment_manifest.json`: Torch2.8/Torchvision0.23, NumPy1.26/OpenCV4.11,
Warp1.6.2, Kaolin0.18, PyTorch3D0.7.9, nvdiffrast0.3.3 and SpConv2.3.6.
Warp1.18 initially required CUDA13-capable driver; Warp1.2.1 enabled GPU but
lacked Kaolin's FEM API. Warp1.6.2 provides both with the existing driver.
No system driver/toolkit/environment changes were made. CUDA extensions
compile with the existing CUDA12.4 toolkit, sm86 and two build workers.

PyTorch3D CUDA KNN, FlashAttention, sparse convolution, nvdiffrast rasterization,
Warp depth kernels, official FoundationPose checkpoint loading and a synthetic
register/track smoke pass. These establish execution only, not real-video
pose quality. Official scorer/refiner assets and pinned vendor commits are
recorded. Hardcoded upstream `/tmp` mesh export is redirected to repository
TMPDIR, with the patch recorded in `upstream/local-output-path.patch`.

Large TRELLIS/SpaTracker/DINO assets completed acquisition by12:55 local. The official
SpaTracker Front model supports206 range responses; its existing prefix is
verified against pinned-source bytes, then resumed with at most four16MiB
ranges. A final official LFS SHA256 is mandatory before publishing the file.
Model stages use local assets only and preserve native camera composition;
learned camera poses are diagnostic outputs. Depth-to-native-hand scale
calibration remains approximate because joints and visible surfaces differ.

Real tape-measure stages: native frames96..143 at30Hz; SpaTracker front
depth passes in27.3s, initial native-hand depth scale0.60736 with176 calibration
samples from the first8 frames, relative IQR0.180. Offline refinement passes
in26.1s with94.99% accepted-depth pixels and native cameras preserved. These
are estimated depth labels, not metric ground truth. TRELLIS seed42/25steps
creates a closed mesh with35714 vertices/71424 faces in135.9s; inspected views
resemble the tape measure, with unobserved geometry still generated.

First actual FoundationPose initialization (`pose-r1`) produced no frames
after242s; only that verified owned process was terminated. Partial metric mesh
and log remain. `pose-r2` adds bounded stack snapshots to distinguish mesh
diameter, mesh-normal/constructor work, model loading and rotation clustering
before changing the algorithm. No useful trajectory or effect window has yet
been claimed from this run.

The30s stack snapshot identifies `compute_mesh_diameter`'s10000x10000
broadcast, before rotation/model initialization. A process-local replacement
in our wrapper computes the exact all-mesh convex-hull diameter in512-row
blocks; vendor estimator and shared code are unchanged. On the same generated
metric mesh this takes0.048s, returns0.063016m, and agrees with independent
full-pair distances for spatial/planar/linear fixtures and rigid transforms.
Six owned contract tests pass in2.16s. Stack instrumentation was removed.

`pose-r3` completes in24.0s; four actual registration attempts on frames96..99
fail silhouette/depth criteria (IoU0.046..0.182), so48-frame processed output
has0 accepted poses/windows. `object_audit.json` truthfully reports
`NO_USABLE_WINDOWS`, preserving source confidence and official-test quarantine.
The attempt counter is corrected to count performed registrations only.
The first frame is already occluded; next bounded decision check uses native
frames64..127 to initialize before grasping, then follow through occlusion.
No quality thresholds are relaxed. This distinguishes an initialization
choice failure from a failure of this particular reconstructed mesh/depth pair.

`pose-r4` finds1 isolated accepted pose, no windows. An implementation audit
then detects scale normalization using a bounding-box diagonal, larger than
the mesh's actual Euclidean diameter. That unnecessarily shrinks geometry.
`pose-r5` uses exact diameter consistently for normalization and estimator:
23 accepted frames, median IoU0.607/depth RMSE0.00796m, but0 continuous windows
because contact occlusion breaks the track. This is an improvement in
engineering behavior, not validated pose accuracy. Runs before this correction
are not evidence against the intended normalized-mesh reconstruction method.
`pose-r6` uses upstream5 registration-refinement iterations and a bounded8
registration budget, preserving the same acceptance criteria; grasp-phase
quality remains intermittent.

Decision Note: before processing more candidates, compare one larger rigid
target (`basic_pick_place/1`, phone) to distinguish small-object/occlusion
difficulty from a broader implementation/reconstruction failure. Evidence:
tape masks/mesh/depth execute and clear-frame silhouettes match, but no0.8s
contact windows pass. Choose a manual phone prompt, mesh from a clear pre-grasp
frame,64 native frames spanning grasp, and identical depth/pose auditing.
Cost<=6min GPU3 within the90min pilot cap, no new weights/data. Stop expansion
if this target also lacks reliable continuous contact windows; retain all
diagnostics and label the engineering-pilot quality UNCLEAR. No external
authorization or global claim change is needed.

## Bounded engineering trial result

The phone control completes all stages:160 native-frame masks, mesh from
clear frame40 (57086vertices/114168faces, closed), depth/refinement48..111.
Front calibration uses176 native-joint observations from48..55, scale0.50824,
relative IQR0.1433. Offline refinement retains94.05% valid depth pixels and
holds supplied camera trajectories fixed. Source inspection confirms
`fixed_cam=True` holds the supplied camera-to-world trajectory, not an identity
stationary camera; SpaTrack normalizes the supplied trajectory to its first
frame and `TrackRefiner` initializes `self.c2w_est_curr` from `cam_gt`.
Camera-space point-map depth remains separate from native origin composition.

`object-pilot/phone/pose-r1` produces57/64 accepted poses and24 overlapping
4-history/24-future windows, anchor source IDs53..76, in63.5s. Accepted-frame
median silhouette IoU0.6624/depth RMSE0.00277m/nearest native hand0.01665m;
these are consistency measures against estimated depth/masks, not real pose
accuracy. Late frames101..105 and other quality failures are excluded rather
than padded. All24 windows have moving category1;0.8s endpoint translation
5/50/95th percentiles0.00341/0.00834/0.14399m. Maximum adjacent displacement
among valid pairs0.04467m and rotation11.86deg. Manual projected-cloud/mesh
views and the native-world replay show plausible early alignment; object
orientation and generated unobserved shape remain unvalidated.

`object_audit.json` passes all24 windows' tensor/collate contracts, exact
canonical provenance, preserved source hand/confidence/clock, native camera
composition and rigid point correspondence (max2.98e-8m). `motion_audit.json`
independently checks the temporal jump filters and records overlapping anchor
identities. `native_world_replay.mp4` includes native RGB/hand projections and
fixed-origin3D geometry; invalid objects disappear, never becoming identity
poses. No official-test labels enter training or checkpoint selection.

This is a lightweight ObjectForesight component adaptation: manual prompts,
direct TRELLIS mesh-decoder vertex colors, no Gaussian UV texture baking/FLUX
enhancement or automatic bulk selection. It is not a reproduction of every
upstream preprocessing stage. Package/weight/source provenance is in
`deployment_manifest.json`; full reproducible paths/parameters are documented
in `docs/REF5_DATA_EXPANSION.md`. Six owned semantic/geometry tests pass.

Decision: retain native40-sequence expansion and the passing single phone
engineering example. The two actual object trials have inconsistent usable
coverage (phone24windows, tape0), so do not start a20–50-clip bulk conversion
on this evidence. Engineering execution is demonstrated; across-object
reconstruction quality remains UNCLEAR. This completes the first bounded
try, not the proposed20–50-clip pilot or any scientific accuracy/utility claim.
Next decision-changing work is clean-frame/occlusion/mesh-quality selection
on additional clearly rigid candidates; source-balanced training integration
and Gaussian texture fidelity remain deferred, with native source sampler
explicitly required before using the existing training launcher.

## Original core follow-up (user requested continuation)

Decision Note: distinguish adaptation-loop limitations from the original
ObjectForesight object tracking behavior on the same tape/phone inputs.
Reuse existing masks, mesh and native-calibrated depth, stage native-rate
video plus EPIC-format object files, then call unmodified upstream
`process_single_object`, initialization/scale selection and Tracker.
Only local imports, sm86 compatibility, bounded estimator diameter and final
mesh capture are adapted; manual clean-frame files remain explicit inputs.
This is an original step10 comparison, not yet the entire ten-stage pipeline.
Start with tape native64..127, phone48..111. Same downstream quality gates
remain IoU>=0.25 and depth RMSE<=max(0.025m,0.15*exact mesh diameter), followed
by original native hand/window checks. No thresholds are relaxed for yield.
Classification: Decision engineering probe; outcomes decide whether to retain
the stock object pipeline and expand toward the ref5 20–50 rigid clip pilot.
Cost<=15min GPU3 per object, within the existing90min inference cap and24GiB
owned-output cap; free reserve>=20GiB. Stop on ownership/resource conflicts,
deadline or nonfinite calibration. No new authorization boundary. Base code
commit at continuation6367d8d0e99c70ff097049b5e13b7e4d0f879ae1; actual adapters
and vendor SHA256 are captured per run.

Original step10 follow-up evidence: tape `upstream-r1` is invalid for quality
assessment because PyOpenGL3.1.0 throws a bytes/string exception. A minimal
64x64 EGL box reproduces it. PyOpenGL3.1.7 then exposes mixed Conda/system
GLdispatch; a process-only system libGLdispatch preload makes the same box
render pass on GPU3. All fixes stay in our isolated environment.
Tape `upstream-r2` actually evaluates five initialization candidates, but
none reaches the stock0.4 initialization IoU gate (best around0.33). This
checks the stock tracking core with the earlier simplified mesh/depth only;
it is not evidence against the full textured ObjectForesight pipeline.

Phone `upstream-r1` reaches the stock8 reinitialization budget, then crashes
on an upstream logging call:17 parameters receive18 positional arguments.
The budget row has10 NaNs for9 metric fields. A minimal AST-extracted
budget-branch reproducer fails before a10->9 correction and passes after it.
`upstream-r2` then finishes tracking but fails debug-video writing: cached
frames have RGB+mask panels while missing frames have RGB only. A two-frame
writer reproducer fails before and passes after adding a blank mask panel
for missing rendered frames. Both source bugs occur in the pinned vendor
step10 file. Runtime text patches preserve that file, all failed runs and
the original pose/scale/acceptance/reinitialization logic. Repro manifests
and patch diffs are in the owned output `upstream/` directory.

Deploy original step8 multi-image reconstruction, Gaussian decoding and
2048px UV texturing next. Additional U2Net175,997,641bytes verifies the
official rembg MD5; domestic model mirror is unavailable, official GitHub
proxy fallback works. PyVista/VTK/PyMeshFix/igraph/diffusers wheels use the
Tsinghua mirror; mip-splatting CUDA extension builds atsm86 with2 workers,
source pin dda02ab5ecf45d6edb8c540d9bb65c7e451345a9. FLUX remains the original
optional fallback when unavailable locally. Cost<=30min GPU3 for one mesh,
within the existing shared90min inference cap/24GiB owned-output cap.
No automatic hand estimation, training or remote publication is performed.

Phone `upstream-r3` completes the original step10 core with the two output
bug fixes,8 successful reinitializations and forward-budget abort. Native
export keeps49/64 quality-accepted poses and17 overlapping windows. The
independent audit passes all17 windows, exact camera composition and fixed
canonical correspondence (maximum2.24e-8m). Median accepted IoU0.600, depth
RMSE0.01114m and hand-to-mesh distance0.02711m are self-consistency measures
against estimated depth/masks, not ground-truth accuracy. The earlier lite
phone run had24 windows; these are different registration/scale algorithms,
not a controlled claim of superiority. Retain both without pooling duplicates.
Six native/geometry tests pass again in2.32s using the read-only graspenv
interpreter (the isolated inference environment does not install pytest).
Original full-texture phone mesh is now attempted using visually inspected
clean native16..39 within selected16..111 (24/96 frames passes stock step8
clean-coverage gate). Source review images are in owned `upstream/`.

Decision Note: the first20 acquired clips include plush/articulated targets,
so they cannot satisfy ref5's20–50 clearly rigid clip selection. Scout30
additional `basic_pick_place` HDF5 metadata files first, then acquire paired
videos only for visually reviewable rigid candidates. Preserve prior20 files
and all manifests. This does not enlarge the20-target reconstruction pilot
or admit official test into training. Acquisition<=2GiB cumulatively for
selected EgoDex files, owned outputs<=24GiB, free disk>=20GiB; no full ZIP
download. Cheap metadata curation changes the target selection decision.

Original step8 completes both phone/tape: default10-image/40-step sampling,
Gaussian decoding, mesh postprocess and2048px UV texture baking. Phone101.75s,
tape105.67s. Optional FLUX is unavailable locally and the original fallback
continues. Phone textured mesh13937vertices/25142faces is retained. The
640x360 phone run `upstream-r4` rejects initialization (IoU best0.337).

Input audit identifies stock morphology/area thresholds in pixel units.
Restore original1920x1080 RGB rather than scaling these gates. Both first
native-resolution runs fail initialization because score preprocessing
backprojects888 candidate crops to1080p simultaneously; a single grid
allocation needs20.58GiB. These are execution failures, not pose-quality
negatives, even though upstream catches exceptions and returns no init.

Memory adaptation: run the original TripletH5Dataset per-pose transform in
32-candidate chunks, concatenate every field in original candidate order,
then execute the unchanged score/selection. No candidate removal, pose/scale
algorithm or acceptance threshold changes. A native1080p33-pose fixture
compares original vs7-pose chunking: every output field is numerically
identical (maximum absolute error0), peak allocation3.15GB. Evidence is
`upstream/native_batch_parity.json`; the runtime adapter SHA is pinned per run.
This supports implementation equivalence for the tested transform, not
scientific reconstruction accuracy. Follow-up native `-r2` runs use this
compatibility adaptation. Accumulated object-stage wall time before these
runs is approximately28min of the90min cap.

Metadata scout30 HDF5 files completes with11,422,548 transferred bytes,
then12 selected rigid-candidate pairs complete with24,987,645bytes. Their
native hand/camera conversion is separately quarantined. Visual review
confirms solid blocks/cubes, plate, cup, bottle body, dice box and rigid toy
screwdriver targets. Existing six additional targets include a white brick,
egg-shaped toy, two small cardboard boxes, bottle and plastic creamer cup;
metadata/rigid approximation and target identities remain explicit.

Native phone `upstream-native-r2` completes candidate evaluation without
exceptions in68.11s after bounded score preprocessing. Best initialization
IoU0.382 remains below the original0.4 threshold: no initialization and no
accepted trajectory. Do not change the threshold to turn this into success.
Full-texture original mesh+tracking has therefore not yet yielded an accepted
clip; the17-window engineering pass used the earlier mesh with original step10.

Decision Note: retain the full20-target engineering pilot authorized by ref5,
but first test the simpler blue cube (`basic_pick_place/6`) to distinguish
thin/small-object geometry difficulty from a general integration defect.
Record20 visually reviewed candidates with input hashes; egg-shaped toy and
plastic creamer retain explicit material/rigidity uncertainty. Stock step8/10
algorithms and quality gates remain fixed. If cube succeeds, proceed serially
through curated targets; if it fails, inspect mask/mesh/depth alignment before
spending the remaining budget. Replace the earlier self-imposed90min object
stage cap with180min cumulative own GPU stage wall time,20 target attempts,
24GiB owned storage and>=20GiB free disk. This stays within the user-authorized
pilot and Campaign's4-GPU/300GB limits; no core claim or external scope changes.
Stop on budget/storage limits or unavailable exclusive GPU. Only known owned
processes may be stopped. No new external authorization boundary is introduced.

Resource recheck: GPU3 acquired a foreign process PID386456 from another
user's splatam environment. Our native phone run has exited. Do not stop that
process or launch new GPU stages alongside it. GPU0/1/2 remain reserved for the
other Ref2Dex agent. CPU contact-sheet curation and input preparation continue.
Next staged work: tape `upstream-native-r2`, then cube6 mask/reconstruction/
native-calibrated depth/original step10. All EgoDex outputs remain test-only.

Input adapter adds explicit native prompt-frame selection for targets initially
hidden in bowls. It calls the installed SAM2 original forward/reverse iterator
and places each result by its returned source frame ID; it does not implement
a new segmentation/tracking algorithm. Runtime GPU verification of nonzero
prompt-frame propagation remains pending until the device is free.

Second metadata scout33..52 completes20 HDF5 files,8,744,838 transferred bytes.
Select red LEGO brick41 and blue dice47;2 paired clips complete with4,266,146
transferred bytes after mirror probe/proxy fallback. Five-frame source review
confirms manipulation without visible body articulation/deformation. Replace
ambiguous egg toy103 and creamer vertical103, retaining their exclusion records
and prior candidate snapshot. Final `pilot20-candidates.json` contains20 distinct
visually rigid-body candidates, not20 accepted object trajectories. Total paired
acquisition is34 clips, plus metadata-only scouts. Native hands for replacements
also convert with the official test quarantine intact.

Cube6 annotation and clean-frame review confirms full cube coverage by box
1096,724,1196,828 and point1144,780 at native frame0, with clean0..20 and
pose-clean10,14,18. `run_original_clip.py` only invokes existing adapters/stages
serially, validates source hashes and checks for an empty GPU before each model
stage. Its first real invocation returns77/RESOURCE_WAIT before model loading.
Foreign PID386456 has ended, but replacement PID417786 from the same external
user immediately occupies GPU3; verified by nvidia-smi and ps. This is the same
resource blocker. Do not retry/resume failed stage directories or overwrite
evidence. The clean prepared cube run can resume after exclusive GPU is free.

Resource blocker audit,2026-10-07 07:25 UTC: the same lack of an idle GPU
persists across3 consecutive goal turns. Last turn made CPU progress through
rigid replacements, native hand conversion and the prepared stage runner.
This turn rechecks all8 GPUs and live PID417786 after a60s observation wait;
the GPU0/1/2 training and GPU3..7 external jobs remain active. No owned model
process is running. Independent inputs for the next decision probe are ready;
its masks/mesh/depth/pose stages require GPU. Mark the continuation goal
BLOCKED_RESOURCE rather than complete. Resume with the documented cube command
after GPU3 is free, then retain the original20-clip scope and perform quality
audit. `resource_blocker.json` records the live-process evidence. Do not kill,
co-run with or change the environment of those jobs to clear this blocker.

User review correction (engineering defects, no reconstruction-method claim):
all3 reported issues reproduce. Full-horizon independent recount confirms546
of2596 old static native windows move above2mm/.02rad mid-horizon (train391,
val99,test56). The previous6-test pass and old ENGINEERING_PASS did not cover
category semantics. Original SE(3)/geometry evidence remains valid, while the
old classification counts are superseded. Native helper now checks all24
future frames with the OakInk motion predicate; native category IDs1/2 are
preserved, and source sampler integration remains pending. Direct minimal
reproductions establish the reported causes; broad hypothesis search is omitted.

The actual pinned choose_best_init loop catches the wrapper's TimeoutError
and processes a second candidate after the alarm. Replace only the wrapper
deadline exception with RunDeadlineExceeded(BaseException); the actual outer
run catch/finally records TIMED_OUT and stops before the second candidate.
Audit_native's external SourceFileLoader attempts bytecode writes under default
settings. Its entrypoint now sets sys.dont_write_bytecode before third-party
imports. Regression intercepts set_data before any write/directory creation.
No external source/cache is modified by these reproductions or fixes.

Four regression cases (translation return, rotation return, actual alarm through
original candidate loop/run boundary, cache write interception) reproduce the
defects and pass after fixes, together with the earlier6 semantic/geometry cases.
Logs: tmp/ref5-data-expansion/review-tests-red.log,deadline-red.log,
review-tests-final.log. No GPU compute or live training change is involved.

Repair creates fresh native3d-r6 by reindexing existing arrays, avoiding MANO
reconstruction. Exactly546 categories2->1 change; all window identities and
linked pose/hand/geometry arrays stay unchanged. Motion/static counts become
24705/2050; total26755 and all split sizes remain unchanged. Per-sequence
motion_windows metadata and source/index/array hashes are regenerated in
category_repair.json. Keep native3d-r5 for provenance and immutable link targets.
Enhanced audit checks all26755 classifications plus original40-source pose/
geometry parity, source hashes, subject splits and WM interfaces; passes with
maximum rigid correspondence error7.30e-8m. Independent EgoDex audit covers
both accepted phone packs (24 and17 windows, overlapping—not pooled) and all
four zero-window tape packs: no motion-category mismatches there. Original
full-texture20-clip pilot and multisource training remain incomplete. No Git
commit/push is made while preserving the other agent's shared worktree.

User authorizes GPU3 retry after the3 engineering fixes. Fresh nvidia-smi
shows GPU3 has no compute process,2MiB/0% utilization. Resume the prepared
cube6 original-r1 through serial adapters with1800s per-clip bound, native
1080p RGB, original step8/step10 defaults and unchanged quality gates. GPU0/1/2
and external GPU4..7 jobs stay untouched. Free disk48GiB is above20GiB reserve.
The next decision remains whether a simpler volumetric rigid target produces
continuous, aligned24-future-frame windows. This is an engineering probe,
not ground-truth reconstruction accuracy or a completed20-clip pilot claim.

GPU3 retry: SAM2 preserves93 source frames and correctly follows cube6.
Original-r1 mesh stage spends653.8s without GLB output, CPU active/GPU0%.
Stop only verified owned mesh PID, preserve raw logs and diagnostic_stop.
This reveals parent control bug: nonzero subprocess exit plus child RUNNING
manifest is incorrectly retained as RUNNING (and a stale COMPLETED child
could be treated as success). Fix stage_status to prioritize exit code and
add6 regression cases; all16 tests pass. GPU0/1/2 tasks remain untouched.

Original-r2 repeats the same original parameters with phase/call-stack capture:
postprocess19.13s, hole filling8.65s, then XAtlas parametrize_mesh stalls on
243359vertices/486850faces. Native stack points to original to_glb line440 /
parametrize_mesh line268. Save mesh_before_uv.npz and exact phase logs.
Stop after222.7s with the issue localized; parent now reports PROCESS_FAILED.

Decision Note: use the original to_glb simplify argument to target50000 faces
before UV expansion on oversized meshes, rather than replacing UV/texturing
algorithms. Default remains original0.7 unless this explicit budget is given;
reconstruction40-step/10-image/Gaussian/texture2048, native hands/cameras and
pose quality thresholds stay fixed. Lower mesh detail changes geometry and
must be recorded with raw face count/effective ratio and reviewed for distortion.
This bounded representation choice is within the20-clip/180min/24GiB pilot.
Retry cube6 original-r3; if it finishes, inspect shape and continue depth/pose;
if still slow, preserve the engineering failure and stop expensive repeat work.
This does not support/refute full reconstruction accuracy or change a core claim.

GPU3 cube6 original-r3 reaches original step10: full textured mesh completes
in152.77s, UV25.08s, texture25.52s; raw1623096 faces simplify to50151
including hole filling. Depth31.92s, refinement12.51s, native staging8.27s.
Original candidate selection accepts frame4 with IoU0.454>0.4; subsequent
tracking/reinitialization is still live and no accepted-window claim is made.
CPU geometry review compares deterministic40000 sampled points per mesh
against retained r2 pre-UV geometry: bidirectional mean0.00367 and95th
percentile0.00686 of normalized bounding-box diagonal (sampling floor included).
Extents stay similar; simplified mesh is not watertight and rare maximum
deviation reaches0.055 diagonal. Retain this geometry limitation; it does not
measure true shape or pose accuracy. Current16 regressions passed, including
parent subprocess exit-state cases. Other GPU tasks are untouched.
