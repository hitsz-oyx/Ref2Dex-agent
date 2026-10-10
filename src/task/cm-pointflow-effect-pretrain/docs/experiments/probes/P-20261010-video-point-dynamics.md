---
schema: ref2dex.probe.v2
probe_id: P-20261010-video-point-dynamics
experiment_id: P-20261010-video-point-dynamics
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: pending
claim_id: C1
hypothesis_family: HF-video-point-dynamics
probe_index_in_family: 1
seed_pool: probe
seeds: [226]
decision_changed_if_positive: expand qualified video tracks and test encoder transfer to native hand-conditioned dynamics
decision_changed_if_negative: diagnose track quality or missing interaction inputs before scaling video training
status: UNCLEAR
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

Pending implementation/qualification at a fixed execution commit.

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
