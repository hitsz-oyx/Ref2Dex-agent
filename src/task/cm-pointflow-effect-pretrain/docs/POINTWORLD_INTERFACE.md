# Ref3 implementation and engineering checks

Implemented on `cm-pointflow-effect-pretrain`, code checkpoint `d1700a3`, using
the user's unchanged local PointWorld clone, commit
`05484826dfef74cbe278a3974179a5a16705d35d`. Exact small blueprint/128dim/128patch,
1cm grid;50,495,881 parameters. The actual count exceeds the old37,208,777;
we retain the requested backbone rather than claiming a parameter reduction.
No visual/uncertainty branch, independent action encoder or dynamics Transformer.
[Design](POINTWORLD_WM24_DESIGN.md), [Probe](experiments/probes/P-20261007-pointworld-small-wm24.md).

All artifacts below live under
`outputs/cm-pointflow-effect-pretrain/pointworld-small-wm24-20261007/`.
The source corpus remains in `outputs/cm-pointflow-effect-pretrain/oakink2-wm30-k24-20261006/`.

Train-only normalization:4096 balanced windows, GPU0, seed216; `norm_stats.json`.
Per-h24 flow std approximately21.41/19.90/20.72mm, translation std
25.81/24.07/25.56mm, rotation RMS0.172rad. Means/stds are embedded checkpoint
buffers and identities include stats, source manifest and sampled-index hashes.
Statistics generation used source checkpoint `f3ba3dd`; later adapter fixes
do not alter sampled windows, feature extraction or computed statistics.

GPU contracts: new point-cap/object-identity test and real-CUDA input isolation,
missing-hand invariance, history action independence, valid rigid matrices,
perfect-target zero loss, nonzero action gradients, finite parameter gradients
and exact parameter restore pass. Together with the existing CPU coordinate,
shuffle, rotation and physical-loss contracts:7passed/1skipped in12.10s. The skip
is the optional OLD-model GPU test requiring WM30_INTERFACE_DATA; the NEW GPU
test ran. Logs: `contract_tests_voxel_fixed.log`.

The first interface check failed repeat-output invariance. Stage hooks localized
differences to the first sparse CPE even with RNG reset and all order shuffles
disabled:7244 raw points vs1788 unique spatial voxels in the inspected batch
(before distinguishing batch IDs). SparseConv duplicate coordinates were unsafe.
Fix: average into unique(batch,voxel) input rows, then restore original point
identities through the inverse map, retaining scene skips and complete labels.
All PTv3 downsampling stages use fixed evaluation orders. Preserve failed test
logs and early smoke files; their outputs are not evidence against ref3.

Corrected repeated-batch learning check: two train moving anchors, IDs
`[52,4,1921]` and `[0,5,1336]`, seed9/80steps. Normalized loss3.01555→0.22490;
anchor h24 EPE44.335→16.382mm.12.30s,peak allocated1.056GB. Artifact:
`learning_smoke_voxel_fixed.json`. This tests optimizer/target plumbing and
small-batch fit only; it is not held-out prediction, causal or policy evidence.

Three independent real-data arms each completed6updates,finite losses/gradients,
and exact saved parameter roundtrip. All share initial parameter hash
`970e6f50deff1e07a743a0c21c4bfaffb92bcf11314bba2fe2cb884aa773ea66`.
Directories `smoke-history`, `smoke-action`, `smoke-shuffle`;6-step configs
override batch/accumulation only for the engineering checks. Held-out TEST
panels remain unopened during engineering; frozen inference uses validation.
Full checkpoint resume loading (including optimizer/RNG) completes with exact
parameter roundtrip; standalone saved-checkpoint validation inference and its
action shuffle pass. Artifacts: `smoke-action-resume/result.json` and
`smoke-frozen-validation.json`.

## Run entry points

Use the existing GPU-compatible `graspenv` Python interpreter. Required installed
packages:torch2.4.1+cu121,spconv,flash_attn,torch_scatter,timm,addict,PyYAML,
numpy. Keep TMPDIR inside the project. No environment or system changes needed.

```bash
CUDA_VISIBLE_DEVICES=0 TMPDIR="$PWD/tmp" python \
  src/task/cm-pointflow-effect-pretrain/tools/run/prepare_pointworld_stats.py \
  --data outputs/cm-pointflow-effect-pretrain/oakink2-wm30-k24-20261006 \
  --output outputs/cm-pointflow-effect-pretrain/pointworld-small-wm24-20261007/norm_stats.json

TMPDIR="$PWD/tmp" python \
  src/task/cm-pointflow-effect-pretrain/tools/run/launch_pointworld_group.py \
  --data outputs/cm-pointflow-effect-pretrain/oakink2-wm30-k24-20261006 \
  --output outputs/cm-pointflow-effect-pretrain/pointworld-small-wm24-20261007 \
  --stats outputs/cm-pointflow-effect-pretrain/pointworld-small-wm24-20261007/norm_stats.json
```

The stats command refuses an existing file; do not rerun it over frozen stats.
The launcher refuses an existing group launch, checks source/stat hashes and
three free GPUs, reserves10GiB disk,monitors every30s and uses one24h deadline.
Each arm saves only bounded best/latest/final optimizer+RNG checkpoints, retains
config/manifests/progress, and stops cleanly with SIGUSR1. Resume using the same
data/config/stats/arm and `--resume PATH` with the single-arm trainer; preserve
original artifacts in a fresh output directory. Frozen evaluator reconstructs
normalization from checkpoint, supports validation engineering inference and
default test inference. No PPO is launched.
