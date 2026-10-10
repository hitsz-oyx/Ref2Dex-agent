# Task state: point dynamics pretraining

Updated2026-10-10. Branch `cm-pointflow-effect-pretrain`; execution on RLG.
Root Mission/Campaign apply: intermediate pretraining must eventually support
action-conditioned Cm and matched self-trained robot policy benefit. Neither
weak video nor raw sensor forecasting replaces that final objective.

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

## Open-source-first follow-up (user steering)

User explicitly requests existing open-source assets before writing new pipelines,
with the decision being whether video helps our existing model. The new
[source usage review](research/2026-10-10-open-video-data-usage.md) distinguishes
author ObjectForesight rigid-pose supervision from our LK pilot, PointWorld
DROID video pretraining, and PointWAM's paper-only human-video recipe.
Pause the proposed custom RGB/LK observation entry and future-sensor expansion.
Offline teacher processing is permissible for representation pretraining, while
deployment/causal forecasting claims still require independently qualified inputs.

Decision note: acquire pinned official PointWorld small-DROID checkpoint
(1,826,853,514 bytes, SHA256 ccb9ed93dff5eea976010c57dd0cb5634db61c68b732c4437cbf54c8da9de8fe)
and inspect backbone key/shape compatibility only. New artifact cap3GiB,
download deadline15min, CPU metadata inspection2min, no GPU training or full
corpus/DINO download. Stop on source/hash drift, collision, cap or incompatibility.
Acquisition completed and strict compatibility passed448/448 tensors with
50,417,280 entries; no key/shape/dtype/finite failures. The
[matched initialization Probe](experiments/probes/P-20261010-open-video-backbone-transfer.md)
is frozen: seed228, two2000-update arms, same native data/stats/draw, random
versus official video backbone, existing trainer, at most two free GPUs and
45min per arm. The initializer integrity tests passed2/2. Native patch128 and
time-aware pooling remain; official patch256 differs. No transfer benefit yet. No new external authorization needed.

## Current decision and resources

Retain native main3 model and all weak-label/sensor evidence. Video follow-up
must separate offline teacher LABELS from independently qualified history/prefix
OBSERVATIONS, then fix a matched benefit/transfer screen before scale-up. No
frontend weights or large new corpus download is justified by these negative
pilot recipes. Broader video/tactile potential remains unproven rather than
refuted; exact generation identity, physical scale, source coverage and cross-
task utility are material limits. Core Mission claim is unchanged.

All training/extraction/inference processes ended; new common-camera inference
9.19s/216.65MiB CUDA and census0.37s CPU. Video artifact group remains about1.9GiB
within5GiB cap; diverse RGB/features/checkpoints about206MiB. No external data
modified. [Windows bridge](../../../../docs/user/连接远程服务器.md) works; NAS has space,
remote GPU driver mismatch remains outside this local campaign. No new push.
