# Task state: point dynamics pretraining

Updated2026-10-11. Branch `cm-pointflow-effect-pretrain`; execution on RLG.
Root Mission/Campaign apply: intermediate pretraining must eventually support
action-conditioned Cm and matched self-trained robot policy benefit. Neither
weak video nor raw sensor forecasting replaces that final objective.

## PointWAM: bounded data engineering pilot

The [PointWAM data card](experiments/probes/P-20261011-pointwam-data-pipeline.md)
records two1-second EgoDex clips processed with native camera annotations,
official CoTracker3 and VGGT-Omega-1B-512. The user-selected ModelScope weights
pass published SHA256 verification and strict official-model loading. Packs
contain scene identities/RGB, ten semantic points per hand, language, explicit
masks,31 positions and30 future displacement targets. Human visual review of
the palm mapping remains pending: six palm offsets come from quantized public
robot viewer geometry with an explicit human-frame convention, rather than an
unreleased unquantized author table. Offline full-clip geometry and partitioned
tracker queries are declared. These official-test samples are engineering-only;
no training, benefit claim or expansion follows from the smoke alone.

## Retained native baseline

OakInk2/GRAB/ARCTIC remain the main sources:4,378,478 overlapping TRAIN anchors.
The repaired sharedstats run completed50000 updates from random initialization;
final endpoint symmetric moving/static object EPE21.896/4.321mm (macro13.109mm).
Selection/population distinctions, frozen identities and limitations are in the
[sharedstats card](experiments/probes/P-20261009-pointworld-main3-sharedstats.md).
Retain final checkpoint under
outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/training-r1/train-action/final.pt.
No new main training, native-transfer or robot-policy claim has been made.

## Video: offline teacher inputs, expansion stopped

[Readiness](experiments/probes/P-20261010-video-data-readiness.md) repaired source
clocks/splits, semantic hand validity/quality and permanent LK loss.34local clips,
20clock files;95candidate H4+K24 hand starts yield ZERO qualified windows.
Keep weak data outside native training. Window births/recovery give a separate
h8 point-only view78train/71dev windows with4/2 supported clips. Fixed500-step
[video learner](experiments/probes/P-20261010-video-point-dynamics.md) loses to
static in both dev clips:24.808 vs9.464 nominal teacher-scale mm macro. Its h24
screen also failed; all initial dev endpoint object labels came from one clip.

New [source contract](research/2026-10-10-objectforesight-spatracker-track-contract.md)
finds public VGGT/tracker processing uses whole video/future information for
historical depth/camera/coords. Historical release generation identity is absent;
strict causal observation inputs are UNQUALIFIED. Our LK2D is history-directed,
but backprojection uses these offline geometry fields. Existing pack-level
future-label mutation tests only establish explicit learner interface isolation.
Old runs remain offline full-clip teacher learnability Probes, with nominal3D
scale; they do not prove deployment history-only prediction or calibrated GT.

Common endpoint-camera scoring atd7f4fff reproduces all original masks/world
errors and shows model pixel errors4.658/31.236 vs world-static2.369/5.303px.
OLS has89.35%/88.76% radial error energy but also loses in projection; purely
radial noise cannot explain the model's transverse errors. Direct image CV can
beat image-static, partly reflecting camera motion, not world-object prediction.
Original100-column tracker queries contain72..95exact unique starts; actual
unique initial object coverage is0..4 per clip. ZERO clips pass the16-point floor
in either split. Do not silently substitute coords, copy duplicate queries or
relax the coverage gate. The video route overall remains UNCLEAR.

## Raw sensor/tactile: usable records, no teacher benefit yet

[Channel contract](research/2026-10-10-egotouch-tactile-channel-contract.md) reproduces
release grids but finds sparse/aliased cells and imputed right-hand channels;
hardware tactile/bend/contact roles remain UNKNOWN. Whole-record normalization
exposes future maxima; later probes use original raw/255 and FIT-only statistics.
Allfalse manual flags do not mean physical no-contact.

[Raw sensor/hand screen](experiments/probes/P-20261010-rawsensor-hand-conditioning.md)
verified10official-TRAIN task records, qualified330/535windows from4fit/5held tasks.
Missing hands remain NaN. Actual future shape groupA h24 macro4.607counts loses
to history4.238/shuffle4.131/persistence3.435. Independent review finds no gate-
changing implementation bug. Conditional future hands are human oracles, not
controllable robot interventions; candidate groupA/B are not calibrated forces.

[Current RGB screen](experiments/probes/P-20261010-egotouch-visual-context.md) verified
ten chest videos38.86MB and original frame count/IDs, max relative clock deviation
0.667ms. Fixed snapshots show interaction but nominal scenario labels often share
one physical table. Frozen ImageNet ResNet18 frame-local current features and six
matched500-update arms also fail: history_rgb4.173/future_hand_rgb4.415 vs
persistence3.435counts. Stop this local recipe; no sensor/native auxiliary expansion.

The processed [pressure-map history Probe](experiments/probes/P-20261011-egotouch-processed-history.md)
then used the released masked `(21,21)` grids with past-only normalization and
lost to persistence on all five held tasks. To answer ref10's next narrower
`(H,A/tau) -> future map` question, the [future-hand action-conditioned Probe](experiments/probes/P-20261011-egotouch-processed-action.md)
added an offline future hand trajectory and a different-task donor shuffle under
the same contract. Pressure-only, past-hand, aligned-future-hand and shuffled-
future-hand all lost to persistence on all five held tasks; aligned and shuffled
MAE differed by at most 0.00099 original-scale units. This closes the current
processed pressure forecast/teacher recipe. It does not reject a calibrated
touch representation, T-Rex F6, RBO multi-interaction data, or simulator CHORD.

## Open-source-first follow-up (user steering)

User explicitly requests existing open-source assets before writing new pipelines,
with the decision being whether video helps our existing model. The new
[source usage review](research/2026-10-10-open-video-data-usage.md) distinguishes
author ObjectForesight rigid-pose supervision from our LK pilot, PointWorld
DROID video pretraining, and PointWAM's paper-only human-video recipe.
Pause the proposed custom RGB/LK observation entry and future-sensor expansion.
Offline teacher processing is permissible for representation pretraining, while
deployment/causal forecasting claims still require independently qualified inputs.

Historical decision note for the mismatched checkpoint-transfer run: acquire
pinned official PointWorld small-DROID checkpoint
(1,826,853,514 bytes, SHA256 ccb9ed93dff5eea976010c57dd0cb5634db61c68b732c4437cbf54c8da9de8fe)
and inspect backbone key/shape compatibility only. New artifact cap3GiB,
download deadline15min, CPU metadata inspection2min, no GPU training or full
corpus/DINO download. Stop on source/hash drift, collision, cap or incompatibility.
Acquisition completed and strict compatibility passed 448/448 tensors with
50,417,280 entries; no key/shape/dtype/finite failures. The
[matched initialization Probe](experiments/probes/P-20261010-open-video-backbone-transfer.md)
is frozen: seed228, two2000-update arms, same native data/stats/draw, random
versus official video backbone, existing trainer, at most two free GPUs and
45min per arm. Initializer integrity tests and both3-update GPU checks passed.
Matched 2000-update runs completed on free GPUs 2/3 using the existing launcher,
Git 199c7db; both arms exited normally at step 2000. This was a checkpoint-
transfer run, not training on the official video data: the training manifest
remained OakInk2/GRAB/ARCTIC and only `small-droid/model-best.pt` was acquired.
The complete transfer result is preserved in
`outputs/cm-pointflow-effect-pretrain/open-video-backbone-transfer-20261010-r1/result_analysis_r1.json`.
Its moving/static gate failed, but the result is `INVALID_IMPLEMENTATION /
INCONCLUSIVE` for the intended official-video-data question. Retain the
checkpoint outputs as recipe-level engineering evidence; do not use them to
reject DROID data or the broader video route.

Correction decision note (2026-10-10): the current question is whether released
video data can improve the native model. The previous intervention used released
weights instead, so it cannot close that question. First perform a bounded
metadata/subset availability audit; do not download the full DROID corpus or
start another transfer run. The published full corpus is reported at roughly
3.9 TB, above this Campaign's 300 GB local artifact cap. Any remote NAS staging,
large subset acquisition or new long run requires a separate resource decision.

The bounded official-data audit is now complete. The official DROID-100 RLDS
shard contains real 320x180 JPEG frames, actions, joint states and language;
it is distinct from the PointWorld-derived H5 annotation package. The first
DROID-100 episode was matched to PointWorld-DROID flow shard `000409`: ten
clips, camera serials `20521388`/`24259877`, 11-step scene flow, gripper pose
and joint fields were restored from the pinned SHA256 package, with finite
sampled values. This establishes data availability and a reproducible pairing,
not a video benefit. No official checkpoint entered this audit and no training
has started. The Task-local adapter audit found that all matched clips have 11
time points rather than the native 4-history/24-future contract, and do not
publish verified point-kind, cross-clip identity or native hand-action fields.
The direct native adapter route is therefore closed; any next Probe must first
choose an explicit raw-sequence, shorter-horizon, or RGB/robot-state contract.
See [DROID-100 video audit](experiments/probes/P-20261010-droid100-video-data-audit.md)
[episode pairing audit](experiments/probes/P-20261010-droid100-pointworld-match.md)
and [native contract audit](experiments/probes/P-20261010-droid100-native-adapter-contract.md).
The separate raw temporal audit found 166 frames at 60 FPS in the matching
official MP4, so raw reprocessing remains technically possible but still lacks
the verified point/hand/action contract. The companion raw H5 files contain
robot/timestamp/camera-extrinsic records and 128x128 RGB, but no depth or
intrinsics; the official processor requires `pyzed`/ZED SVO access, which is
absent here. Do not install system dependencies or infer calibration silently.

Correction after checking the author data/training repositories: the raw
reprocessing boundary above does not close the released-data route. PointWorld's
fixed release contract is one context frame plus ten prediction steps, exactly
the 11 time points in these generated H5 clips. A local official-pipeline smoke
accepted all 10 matched clips, produced an 8/2 train/test WDS split, and the
official scene decoder recovered both cameras with finite 11-step flow, RGB,
depth, intrinsics, extrinsics, visibility and depth-valid masks. This used no
official checkpoint and no ZED. The direct native `H=4,K=24` adapter remains
`UNPROMISING`, but a short-horizon `K_aux=10` auxiliary pretraining route is
open; its next required experiment is a random-initialization control that
uses the same official H5/WDS data and transfers only the shared representation
into the native downstream task. See the [source-level reuse note](research/2026-10-10-droid-no-zed-reuse.md).

The matched [metric-3D motion-token Probe](experiments/probes/P-20261011-video-motion-token.md)
also completed. A DynamicWAM-style global history token was wired as an
auxiliary path while preserving the native pointwise history and Huber flow
loss. Background behavior passed the gate, but the two qualified object clips
gave 13.726 mm motion-token H8 EPE versus 12.949 mm matched control EPE, with
both clips no better than control. Mark this particular token recipe
`UNPROMISING`; keep the official-data K_aux=10 route and calibrated
wrench/geometry routes open. This does not establish a negative result for
video data as a whole.

The first actual data-only [official H5 auxiliary Probe](experiments/probes/P-20261011-official-video-aux.md)
also completed from random initialization, without the published model
weights. Its one-context color/normal-only input lost to zero-flow persistence
at H10 (2.263 vs 1.878 mm) on the two held tail clips. Keep the route open for
the explicitly separate H5 gripper/joint-state-conditioned auxiliary; this
result does not answer whether official video plus robot state can help the
native representation.

The follow-up [official H5 state-conditioned Probe](experiments/probes/P-20261011-official-video-state-aux.md)
then used only context joint/gripper state in the same 18-D input and the
pre-registered seed 231. It also lost to persistence on held H10 (2.183 vs
1.878 mm), so close this one-context H5 recipe for further transfer investment.
Keep the conclusion local: the episode/split is small and this does not reject
other qualified official RGB/state front ends or measured-wrench routes.

The direct [official-video data-only native-transfer Probe](experiments/probes/P-20261011-official-video-native-transfer.md)
then trained a native-compatible normal-only H5 auxiliary from random
initialization and copied only its shared scene projection/PTv3 core into a
matched native action arm. The 500-update balanced H24 panel was 101.580/6.020
mm moving/static for the video-data arm versus 101.296/5.977 mm for random;
macro 0.053801 versus 0.053637. The result is `UNPROMISING` for this explicit
feature mapping and short transfer recipe. This closes the current data-only
initialization route for further investment, while retaining the official H5
adapter as a valid data contract and keeping the conclusion separate from
other RGB front ends, T-Rex, RBO multi-interaction data, and simulator CHORD.

The bounded [RBO RGB-D/FT adapter Probe](experiments/probes/P-20261011-rbo-wrench-alignment.md)
then completed on one pinned 42 MB interaction archive. All 295 RGB anchors
matched registered depth, FT, joint, rigid-body, camera, object, and sensor TF
streams within 16.00 ms maximum (2.70 ms median), with finite sampled points
and geometric object-frame wrenches. Keep RBO open for a matched geometric-
wrench auxiliary Probe, while retaining the explicit
`payload_compensation=false` boundary; this is not yet CHORD object-COM net
wrench evidence. The follow-up [RBO measured-wrench auxiliary
Probe](experiments/probes/P-20261011-rbo-wrench-auxiliary.md) used a
chronological 220/64 window split and a random-initialized MLP. Both raw
sensor and geometric wrench forecasts lost to persistence at H8 (3.61x and
2.83x L2, respectively), so close this one-sequence forecast recipe while
retaining the adapter contract. This is a local negative learnability signal;
it does not reject cross-interaction RBO training, T-Rex fingertip F6, or
simulator-side CHORD support.

Decision note (2026-10-11): the current choice is to stop expanding this
single-interaction RBO forecast and preserve the verified adapter for a future
multi-interaction matched arm. The key evidence is the complete timestamp/frame
contract plus the held-tail loss to persistence. Cost is minimal and no new
resource boundary is crossed; further RBO investment requires additional
interactions and an object/task split, while the present Mission claim remains
unchanged.

## Current decision and resources

Retain native main3 model and all weak-label/sensor evidence. Video follow-up
must separate offline teacher LABELS from independently qualified history/prefix
OBSERVATIONS, then fix a matched benefit/transfer screen before scale-up. No
frontend weights or large new corpus download is justified by these negative
pilot recipes. Broader video/tactile potential remains unproven rather than
refuted; exact generation identity, physical scale, source coverage and cross-
task utility are material limits. Core Mission claim is unchanged.

Earlier video/sensor training/extraction/inference processes ended; the paired
open-backbone transfer runs are also complete and bounded. Common-camera inference
9.19s/216.65MiB CUDA and census0.37s CPU. Video artifact group remains about1.9GiB
within5GiB cap; diverse RGB/features/checkpoints about206MiB. No external data
modified. [Windows bridge](../../../../docs/user/连接远程服务器.md) works; NAS has space,
remote GPU driver mismatch remains outside this local campaign. The user pause
was lifted before this result analysis; the original pause record is retained in
`outputs/cm-pointflow-effect-pretrain/open-video-backbone-transfer-20261010-r1/pause_status.json`.
No new push.
