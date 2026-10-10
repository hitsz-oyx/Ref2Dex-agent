---
schema: ref2dex.probe.v2
probe_id: P-20261010-video-point-dynamics
experiment_id: P-20261010-video-point-dynamics
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 848be0993d80111ba2a9f6bc22008267b842429e
claim_id: C1
hypothesis_family: HF-video-point-dynamics
probe_index_in_family: 1
seed_pool: probe
seeds: [226]
decision_changed_if_positive: expand qualified video tracks and test encoder transfer to native hand-conditioned dynamics
decision_changed_if_negative: diagnose track quality or missing interaction inputs before scaling video training
status: UNPROMISING
run_id: video-point-dynamics-20261010-r1
---

# Video point prediction without dense future hand labels

Class: Decision. The global goal remains action-conditioned Cm for self-trained
robot manipulation; video history prediction is an intermediate representation
pretraining test, not a replacement for action conditioning or policy validation.

Question: Does the existing PTv3 encoder plus a point-displacement head learn
held-source-video motion information from weak RGB/depth tracks beyond static
and constant-velocity baselines? A positive result justifies expanding video
labels and testing transfer into native geometry training; a negative result
blocks blind scaling and prioritizes label/interaction-input diagnosis.

## Design / decision note (pre-run)

Dense Contact hands have zero accepted local H4+K24 windows. Choose a separate
video-only loader/head that never reads Contact, FoundationPose, TRELLIS, future
hands or tactile targets. Retain the existing native model/data/checkpoints.
Reuse PTv3-small topology and the native 18-component scene feature meanings;
missing surface normals are explicitly zero. Predict per-point future flow
rather than pretend the clip has rigid-object GT. Train from random
initialization; a transfer comparison is a later Decision gate.

Data: already-local ObjectForesight shards, require clock metadata and official
TRAIN object membership. All P03_03 clips are eligible for TRAIN selection;
all P03_13 clips are held for development only. P01_03 missing-clock clips are
excluded. Choose up to4 nonoverlapping clips per video in stable sorted order;
require55..600 source frames, no ambiguous camera convention (>=5x static
reprojection separation). Source videos and selected action time intervals may
not overlap across partitions. Development is a held subset of original TRAIN
material, not an official test score. Contact-test frames are never training
inputs. If qualification leaves fewer than2 clips per partition, do not train.

Tracks: fixed source stride2, record actual29.97002997Hz and original frame IDs.
RGB LK requires forward/backward error <=1 pixel and object mask membership;
failed IDs never revive. Background is frame-zero metric depth, with per-frame
depth/mask occlusion checks; retain measured depth jitter, not forced zero flow.
Initial track candidates are selected from frame-zero observations only. For
each H4+K24 window, input points are selected using all four HISTORY validity
frames only, with up to128 object and128 background points. Future label masks
may affect loss/metric support but never input point selection or features.

Features: history velocity/acceleration use actual timestamps, history
displacement and static indicator use history only; anchor frame is a fixed
current-point-derived center, independent of future labels. Normalization uses
TRAIN inputs only. Future flow loss uses a fixed0.01m Huber scale; object and
background losses each receive half the mass when both are present. Each
window/group is averaged before pooling windows, so abundant background or
longer valid tracks cannot swamp object dynamics. All invalid target entries
are excluded before arithmetic. Report supported counts and physical-metre EPE.

Controls: zero-flow/static and constant velocity from the last two history
frames at actual elapsed time. Forward receives xyz/features/input validity
only; future labels/masks, point kind and split identifiers are absent from its
interface. Test future-label changes leave model inputs and outputs unchanged,
and invalid labels do not affect loss. Verify CUDA backward/checkpoint roundtrip
before a non-smoke run.

Training: one currently free GPU, seed226, batch2, max500 updates, AdamW lr1e-4,
weight decay.01, grad clip1, BF16 where supported, fixed final checkpoint only.
Development evaluation every100 updates is diagnostic; no threshold tuning or
best-step selection. Decision criterion: final development object-track h24 EPE
improves >=10% over the better of static/CV, while background h24 EPE is <=1.2x
static. Report all-horizon EPE as well. No scientific conclusion from training
loss or a fixed-batch overfit; results are Probe labels only.

Budget: CPU preparation<=20min, one GPU<=15min, new artifacts<=5GiB total,
raw extraction<=2GiB. Stop on nonfinite data/loss, source/hash drift, collision,
GPU conflict, resource cap or deadline; preserve failed run/log. Inspect GPU
utilization/memory after startup and estimate ETA. No system modifications,
remote training, full download, existing checkpoint overwrite or new branch.

Outputs: `outputs/cm-pointflow-effect-pretrain/video-point-dynamics-20261010-r1/`.

## Results

Both fixed500-update model probes and the support/fit audits completed;
run-level gates failed. See the recorded execution commits and results below.

The first preparation invocation at `2d8a573` failed before any source extraction:
relative `__file__` was passed to absolute Task-relative identity logging. The
empty `data/` output is preserved with a failure note. Both preparation/training
entry paths are resolved before hashing; the replacement output is `data-r2/`.
This startup failure provides no scientific evidence about video learning.

The fixed500-update run completed at `848be09`: development object h24 EPE
21.861mm vs static18.294mm and CV88.943mm, so the fixed scale-up gate fails.
Before interpreting this as a method result, run a bounded checkpoint diagnosis
of train/development EPE per clip with identical baselines, plus a current-only
input intervention (zero historical kinematics/static indicator). No model
training or step selection. Budget<=90s on one free GPU within the same15min
group budget. The intervention can expose dependence but is not a matched
training comparison. Request an independent read-only semantic review under
AGENTS14; preserve the original negative result and all outputs.

## Limitations / future evidence

Two source videos and overlapping temporal anchors cannot establish large-scale
or independent-episode generalization. LK consistency is not ground-truth
physical correspondence; static depth jitter remains label noise. Future
visibility masks condition scored support and must be reported. History-only
predictability does not demonstrate human-action causality or robot usefulness.
Positive evidence still needs native encoder-transfer and matched conditioning
controls, then the global Mission's self-trained robot policy comparison.

## Completed diagnosis and next bounded decision (2026-10-10)

`train-r1/fit-audit.json` at `6663d42` reproduces the negative development
result. On the two train clips with h24 support, model/static object EPE is
4.70/7.51mm and2.86/5.64mm. Current-only input intervention changes development
object h24 only21.861→21.875mm; this is not a matched conditioning experiment.
Independent read-only review found no definite indexing, future-input leakage,
loss or checkpoint-identity error. Label quality remains weak.

Crucially,28 training windows provide only150 scored h24 object observations
(125 from `_14`,25 from `_15`). All4034 development h24 object observations
come from one clip, `P03_13_12`; `_10/_19` supply none. Windows overlap and are
not independent episodes. Run-level result is UNPROMISING at the preset gate;
route-level evidence remains UNCLEAR. Do not scale from this result.

Decision: distinguish premature clip-start track death from intrinsically poor
long-horizon support. Reuse exactly the same seven clips, source hashes, split,
LK F/B1px, masks, camera convention and fixed training recipe. Initialize new
object/background identities at EACH window's first HISTORY frame, then keep
identity stable for its55 source frames; failed RGB identities never revive
within a window. Separate births do not assert cross-window identity. Enumerate
all stride2 starts, keeping windows/points by HISTORY support only. Report
future support separately; do not use it to choose input points. Qualification
for another500-update run requires >=16 scored h24 object observations in at
least2 clips in EACH split. If this corpus gate fails, stop training and inspect
motion-support/label alternatives. New output `data-window-births-r3/`; preparation
<=600s CPU, <=1GiB, existing total5GiB/15min GPU limits apply. This is a reversible
sampling refinement within the existing Mission and user authorization.

The `ed531f3` window-birth audit completed29.60s, writing12.74MB. Object h24
support by train clip: `_12`223, `_13`0, `_14`122, `_15`32. Development:
`_10`0, `_12`3705, `_19`0. Three train clips qualify but only one dev clip;
status `INSUFFICIENT_LONG_HORIZON_SUPPORT`. No new training launched.

Next cheap step: recover only previously extracted rejected clips from the
same source videos in fixed sorted order. Check nonoverlap, official TRAIN
object membership, source bytes against archive, rigid camera and >=5x convention
separation again. Reuse window births and the same support gate without changing
thresholds. No new dataset download or source-video split. Output
`data-recovered-births-r4/`, CPU<=600s, total artifact cap unchanged. If a second
dev clip still fails, choose a shorter-horizon/data-quality probe before training;
large-scale training remains gated.

The `fbcbc07` recovery adds `_8` with6 history-valid dev windows but no h24
object support; `_17/_4` still have no accepted history windows. The same
h24 corpus gate fails. Preserve this result; no h24 retraining or scaling.

Short-horizon follow-up protocol (before training): qualify h8 (actual0.26693s)
from `data-recovered-births-r4/`, >=16 scored object observations in >=2 clips
per split. All history-qualified windows/points remain present; future support
only qualifies the corpus and determines reported metric support. Retain all24
labels for audit, but h9..h24 are excluded from training loss and scored metrics.
Use same random initialization/seed226, PTv3/head topology, batch2/AdamW and
fixed500-update final checkpoint; do not reuse/select the negative trained model.
Result addresses short-term motion only, not long-term prediction.

Fix the decision gate before execution: model object h8 clip-macro EPE over dev
clips with>=16 labels must beat the macro of each clip's better static/CV by>=10%,
with no qualifying clip worse than its better baseline by>20%, and background
point-weighted h8 EPE<=1.2x static. Report each clip and support alongside pooled
metrics; sparse clips are still reported and trained according to their original
split, not silently removed. Diagnostic evaluations cannot select a checkpoint.
Negative means investigate label/interaction quality before additional training;
positive justifies a bounded native transfer comparison. Output view
`data-h8-view-r5/`, run `train-h8-r2/`; one verified-free GPU<=15min cumulatively,
all outputs<=5GiB. Automated NVML utilization/memory samples supplement ETA and
allocator-peak telemetry. No full dataset acquisition or remote machine work.

## Short-horizon results and attribution boundary

Qualification at `6d7fdb2`:78 TRAIN and71 development overlapping windows;
4 TRAIN and2 development clips meet the preset h8 support floor. The fixed
500-step run completed96.43s with exact checkpoint roundtrip. Peak CUDA model
allocation1001.08MiB; sampled GPU1 memory1367–1389MiB, utilization11–51%.
The small batch/point count leaves GPU capacity unused; this is a cheap Decision
Probe, not a throughput benchmark. GPU0 belonged to an unrelated job throughout
this run. ETA converged to about90s after warmup and all worker compute ended.

| Qualified dev clip | h8 labels | model EPE mm | static mm | CV mm |
| --- | ---: | ---: | ---: | ---: |
| P03_13_12 |3941|11.002|6.763|28.772|
| P03_13_19 |309|38.614|12.165|42.441|
| Clip macro |2 clips|24.808|9.464|35.606|

Sparse dev `_10`6 and `_8`4 scored object observations remain reported in
`train-h8-r2/final-development.json` and are not used for the predefined macro.
Background h8 EPE6.723mm versus static6.233mm passes the1.2x constraint;
object macro fails the>=10% benefit gate. Result UNPROMISING for this recipe,
route still UNCLEAR. Diagnose train fits/label quality and independently
recheck h8 loss/metric support before any more model training; no expansion or
native transfer is authorized by this negative signal. Fit diagnosis<=90s,
no updates; reuse the model and exact data identity, all outputs preserved.

The h8 fit audit at `75d5f4c` confirms large train/dev motion differences:
train `_12` model/static78.800/117.483mm and `_13`49.376/64.461mm;
near-static `_14/_15`3.496/3.210 and4.121/3.397mm. The model is learning
some training motion but has not demonstrated held-video gains.500 steps,
source shift and pseudo-label noise remain alternative explanations; missing
hand context is not established as the cause. Independent CPU review reproduced
clip support/static/CV and macro to1e-8m; real gradients h9..24 were exactly0,
future mutations left inputs unchanged, and a true cross-clip batch retained
correct metric ownership. No defect overturns the short-run negative result.

Engineering note: the initial3-update smoke was inadvertently launched onGPU0
while an unrelated process had occupied it after an earlier check. It finished
in9.70s before an ownership-checked attempt to stop our worker. No unrelated
process was touched. All substantive training/fit runs used freshly inspected
freeGPU1. Future launch checks must be inspected before dependent startup.

Next Decision: audit pseudo-track physical consistency and input motion noise;
qualify tactile-only channel masks before any auxiliary loss. Large training and
native-transfer comparisons remain deferred until a useful minimal signal.

Before the next file-only diagnosis: compare last-two-frame CV with four-history
mean and exact four-point OLS velocity, all derived exclusively from observed
kinematics/displacement at the actual uniform clock. Reuse h8 target support and
report each clip's object/background EPE. Class Decision: if history OLS beats
static by>=10% in BOTH qualifying dev clips, there is an observable baseline
signal to justify checking model optimization; otherwise retain label/input
quality as the next gate and do not add training. This is not a physical-noise
measurement because the targets themselves are pseudo-labels. CPU statistical
arithmetic only, <=30s/100KiB, output `history-motion-audit.json` in the group.


The history-only arithmetic audit at `a0ec3d5` completed0.312s CPU. h8 dev
object OLS/static EPE is15.014/6.763mm for `_12` and20.523/12.165mm for `_19`;
no preset observed-motion signal. OLS improves the noisy last-two CV
28.772/42.441mm but still loses to static in both qualifying clips. This supports
history/target quality inspection before another optimization run; it does not
identify true physical noise or prove that history contains no usable signal.

## Rigidity consistency decision (before execution)

Class Decision: after the failed neural and history-only baselines, ask whether
weak object3D correspondences admit a single rigid transform or are internally
inconsistent. This can distinguish a label-quality priority from another model-
optimization attempt; it cannot establish true reconstruction error.

On exactly the h8 view, use the same HISTORY-selected object IDs and original
future validity. Require>=8 endpoint correspondences and current centered cloud
second singular value>=1e-5m. Fit a proper SE(3), fixed scale1, using endpoint
labels only; this is explicitly an oracle diagnostic, never a forecasting control
or new target. Report point-weighted residual/static EPE and support per clip.
If both qualified dev clips have residual/motion>=.5, prioritize pseudo-label
consistency; if both<=.25, prioritize predictive conditioning/optimization.
Other cases remain UNCLEAR. Thresholds do not remove training windows or points.
CPU arrays/SVD<=30s, <=100KiB, no GPU/model/new download; output
`rigidity-audit.json`. Retain raw labels and all previous results.
