# WM30 implementation and real-data interface checks

Complete implementation follows [user architecture](user/架构.md) and
[confirmed data/model contract](WM30_DESIGN.md). No Policy/Evaluator/Y, KNN,
RGB or contact head. Full width384,8scene/4action/6dynamics layers,6heads,
SparseConv64/128/256;37208777parameters. Full training starts from the same
seed211 initialization in each arm, not from any PPO or smoke checkpoint.

## Entry points

- `src/oakink_wm/data.py`: fixed-current-frame data and masks,0.5m local
  selection using transformed mesh surface centroids, analytical rigid targets.
- `src/oakink_wm/model.py`: sparse scene encoder, semantic action encoder,
 24 queries/object and9D head (translation3/rotation6), rigid correspondence loss.
- `tools/run/prepare_oakink2_wm30.py`: verified-arrival streaming MANO/30Hz
  conversion, continuity filtering, canonical normals and sequence splits.
- `tools/run/train_oakink2_wm30.py`: history/action/shuffle training, physical
  losses, natural/balanced validation, optimizer/RNG checkpoint resume.
- `tools/run/evaluate_oakink2_wm30.py`: frozen-checkpoint held-out test evaluation.
- `tools/run/launch_wm30_group.py`: full-corpus gate, three independent GPU
  workers, shared deadline, failure propagation and test evaluation.
- `tools/audit/audit_wm30_splits.py`: disjoint sequence splits, object/prefix
  overlap and hashes of processed inputs.
- `configs/wm30_k24.json`: matched hyperparameters and resource limits.

## Engineering checks completed

Use12 already-downloaded sequences,123.72s preparation on GPU0;8train/2val/2test
sequences and64416/38976/15242eligible windows. Corpus-wide627 preparation is
separate and does not reuse this subset's split indices or trained weights.

Six contract tests check current-frame analytical correspondence/global-frame
invariance, geometric-center neighborhood selection, valid6D rotations with
finite identity gradients, whole-chunk same-hand shuffle/masks, no GT-effect
access in forward/masked-action invariance/action and sparse-stem gradients,
and physical vector-L1/radius-scaled losses. GPU atomic reductions require
1e-6 numerical tolerance for repeated-forward equality; GT access is separately
forbidden with an input mapping that raises on label lookup.

All three arms complete six real-data optimizer updates with finite loss and
gradient; initialization SHA256 is identically
36633e84352a2304139768fcb7f7955ceeb087c826fd81b2d2a76dd82880d9b0.
Saved checkpoint parameters match memory exactly. Resume restores model,
optimizer,step and RNG; independent checkpoint evaluation works on separate
sequence-held-out engineering samples. These establish interfaces only.

Artifact root: `outputs/cm-pointflow-effect-pretrain/wm30-interface-20261006/`.
Keep original origin-based cache/smokes and failure log; a review found some
mesh origins are>20cm from geometry centers. Geometry-center selection fixes
that0.5m neighborhood error; `processed-origin-v1/` preserves prior extraction.
Current caches use schema `oakink.wm30.geometry-centers.v2`. Earlier smokes used
coordinate-mean L1; final loss matches user vector L1 and is separately tested.
Final action smoke reruns six updates with this loss. All retained numbers are
engineering-only and must not be interpreted as model-quality comparisons.

## Full run status and budgets

Full corpus root: `outputs/cm-pointflow-effect-pretrain/oakink2-wm30-k24-20261006/`.
Download632 checksummed files (627annotations +5assets); hardlinks reuse
existing files, HTTP byte-range resume handles interrupted transfers. Preserve
failed/interrupted manifests and logs. Acquisition<=40GiB raw; additional
artifacts<=70GiB,10GiB disk reserve, preparation<=10800s.

Training group: GPUs0/1/2, at most4GPUs including GPU3preparation. Shared24h
wall deadline from group training start;40000updates/arm, microbatch2×8gradient
accumulation=16windows/update,640000draws/arm. This upper bound uses measured
full-architecture smoke (~0.1s/microbatch); observed runtime may differ on the
full corpus. Stop earlier at fixed updates, nonfinite values, resource conflict,
input drift or group deadline. A budget-stopped unequal-update comparison is
not accepted as a matched scientific result. Latest/best/final are bounded
run-owned files; external checkpoints are untouched.

Frozen panels report horizons1/4/8/12/24, translation/rotation/pointEPE and
center error for anchors, moving objects and full local scene. Natural and
balanced panels include static baseline; H+A additionally evaluates shuffle.
Test sequence windows are not used for fitting or checkpoint selection. Shared
objects/scene/task prefixes are reported, not described as unseen generalization.
