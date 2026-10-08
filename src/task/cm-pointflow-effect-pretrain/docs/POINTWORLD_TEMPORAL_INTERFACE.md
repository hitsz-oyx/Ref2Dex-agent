# PointWorld temporal-preserving correction

## Fixed workspace and static-object selection repair

The temporal adapter shares the fixed anchor-frame voxel contract from
`POINTWORLD_INTERFACE.md`: `grid=floor((x-anchor_origin)/0.01)` with origin
`[-1,-1,-1]`, independent of batch partners and future action points, and
fail-fast bounds `0..65535` on every valid coordinate. The evaluator reports
the existing moving-object mask and its strict complement `static_objects`
(translation `>2mm` or rotation `>.02rad` at any horizon). New checkpoints use
the fixed selection score

```text
(model/anchor/cat0/h24/point_epe +
 model/static_objects/cat-1/h24/point_epe) / 2
```

while preserving both raw components in validation records.

User ref4 diagnosis established substantial action time merging, and the user
subsequently requested implementation. Keep the running V1 three-arm sources,
configs, statistics and checkpoints frozen. This revision is a separate model,
config and train/evaluate entry on the existing branch. It changes neither the
Mission claim nor the data/split/normalization contract. No new long training
is launched by this correction.

## Chosen implementation

Use the same PTv3-small weights topology, parameter count and spatial grid.
The adapter aggregates by `(sample, time, x, y, z)`, with scene time0 and action
times1..24. Every encoder GridPooling level includes time in its cluster key.
Spatial3D SparseConv CPE uses virtual batch `sample*25+time` to retain unique
indices. Serialization, attention batch and offsets retain the physical sample,
so scene and future times still interact in attention. This avoids invalid
repeated SparseConv coordinates and avoids isolating time in attention. The
upstream submodule remains unchanged. Within-time spatial merging is allowed.

The action summary is max over hands/keypoints separately for each horizon,
and feeds the corresponding object/horizon head. The current scene skip and
full512-point rigid supervision remain intact; no future labels enter forward.
A global max over all action timestamps no longer supplies the output head.

For the default cumulative selector, define at each horizon and surface point:

```
r = max(norm(p_future - p_current)/0.002, cumulative_rotation_angle/0.02)
w = 0.1 + 0.9 * sigmoid(5 * (r - 1))
```

Apply the valid-object mask before normalization, and normalize weighted point,
translation and rotation losses as before. Static supervision retains a positive
floor; slow cumulative displacement and small-object rotation can receive high
weight. The2mm/0.02rad scales follow the existing moving-window thresholds;
these are a declared recipe choice, not empirically optimized thresholds.
The surface criterion need not reproduce the sampler's object-origin label.
`motion_weighting=released_incremental` retains the exact original selector
and normalized losses as an ablation. Raw weights alone are not gradient shares.

Rejected alternatives: changing only the entry key would still merge time at
later pooling; setting the attention batch to time would remove unified
scene/action interactions; shifting coordinates by time distorts physical space.
The SparseConv CPE itself now works within a time slice, while attention performs
cross-time mixing. Increased token counts can increase runtime and memory.

## Entries and identity

- [Model](../src/oakink_wm/pointworld_temporal.py)
- [Configuration](../configs/pointworld_temporal_wm24.json)
- [Trainer](../tools/run/train_oakink2_pointworld_temporal.py)
- [Evaluator](../tools/run/evaluate_oakink2_pointworld_temporal.py)
- [Regression contracts](../tests/test_pointworld_temporal.py)

Temporal checkpoint identity records all reused and new model/data/trainer/
evaluator sources and upstream sources. Resume and independent evaluation reject
implementation drift. The temporal entry rejects legacy V1 configs/checkpoints.
Use a fresh output directory; never resume a V1 checkpoint as a temporal model.

## Engineering verification protocol

CPU tests exercise the collapsed-voxel reproduction, all four time-preserving
pool levels, invalid object padding, static floor,40mm/24frames slow motion,
zero-radius rotating objects, exact released-selector loss parity and config
rejection. CPU is appropriate for these tiny label/key checks.

GPU3 is free at launch; V1 uses GPUs0/1/2 and unknown GPUs4..7 are untouched.
Run real train-window inference, BF16 forward/FP32 loss/backward, time identity
and unique indices at every stage, repeatability, masks, no-label forward,
action-time sensitivity, rigid rotations and checkpoint roundtrip. Then run
three updates plus checkpoint resume and independent val-only evaluation.
This is engineering smoke, not a research comparison: no quality claim, no test
labels, no threshold tuning. Budget: one additional GPU, at most15minutes,
additional bounded outputs<=4GiB, shared four-GPU/storage limits unchanged.
Stop on nonfinite results, identity drift, resource conflict or deadline.

GPU test command (graspenv):

```bash
TMPDIR="$PWD/tmp" CUDA_VISIBLE_DEVICES=3 \
POINTWORLD_DATA="$PWD/outputs/cm-pointflow-effect-pretrain/oakink2-wm30-k24-20261006" \
POINTWORLD_STATS="$PWD/outputs/cm-pointflow-effect-pretrain/pointworld-small-wm24-20261007/norm_stats.json" \
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q \
  src/task/cm-pointflow-effect-pretrain/tests/test_pointworld_temporal.py
```

Smoke invocation uses the new trainer with `--smoke --steps 3 --arm action`,
the same DATA/STATS, the temporal config, a fresh run directory under
`outputs/cm-pointflow-effect-pretrain/pointworld-temporal-smoke-20261007/`,
and an absolute deadline. Resume is checked in a separate output subdirectory.
Independent evaluator is restricted to `--split val --engineering-only` here.

Predictive quality and the benefit of cumulative weighting require a later
matched comparison. The current
V1 run remains a control; its raw training loss is not directly comparable to
this different weighted recipe.


## Executed engineering results

Model correction commit `eb47ae8`; relative-script identity correction `c16d00e`;
final train/evaluate numerical-settings commit and smoke runtime `616d413`.
Final artifacts are in
`outputs/cm-pointflow-effect-pretrain/pointworld-temporal-smoke-20261007/`,
with the final checks in `verification-r2.json`. Source/stat/vendor hashes are
recorded by the new trainer; every original V1 monitored source remains unchanged.

The GPU semantic suite passed13tests with1unconfigured historical WM30 GPU
contract skipped. Six CPU regression tests subsequently passed, including the
relative invocation identity and shared numerical-settings check. The24action
records plus scene in one spatial voxel now produce25distinct temporal tokens;
all four encoder pooling levels preserve time identity. Real CUDA hooks verify
unique SparseConv indices and all25time identities per physical sample, with
physical attention offsets preserved. Masks, future-label exclusion, action
record/time sensitivity, history isolation, rigid rotations and finite BF16
forward/FP32 loss/action-and-time gradients passed. Parameter count stays
50,495,881; the initialized parameter hash equals the original V1 hash.

The final new-entry action smoke completed3optimizer updates in8.93seconds
including validation/checkpoint work (see its actual result.json for elapsed
seconds). Checkpoint model roundtrip is exact. A separate resumed invocation
restored step3with exact model and all474optimizer states; it does not add an
optimizer update and is a load/lifecycle check. Independent evaluation opens
12validation samples per panel, no test labels. Trainer and independent
evaluator share TF32 settings. Metrics are within engineering numerical
tolerance: max point EPE difference0.03142mm, translation0.00387mm and rotation
0.000226rad. They are not bitwise-identical across processes; no scientific
quality claim is inferred from the tolerance or the short smoke's metrics.

The first relative-path invocation failed before training; its log/config are
preserved. An earlier smoke identified differing train/evaluator TF32 defaults,
now unified. Its final checkpoint/results remain; duplicate engineering-only
best/latest/resumed checkpoint copies were pruned after exact state verification
to respect the4GiB bound. The final run uses fresh `train-action-r2` and
`resume-action-r2` directories. GPU3 is released after the checks; original V1
GPU0/1/2 workers continue under their unchanged group budget.

Parallel-agent coordination is in `tmp/pointworld-temporal-agent-ownership.json`.
This agent owns the listed temporal correction files and this document only;
shared README/STATE/index and another agent's ref5 file are left to their owners.
Future complete three-arm temporal training needs a frozen matched protocol;
use the temporal train/evaluate entries linked above. The existing V1 launcher
continues to run its original model and is not a temporal launcher.
