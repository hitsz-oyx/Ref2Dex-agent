# PointWorld temporal-preserving correction

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

Results will be recorded after executing the checks. Predictive quality and the
benefit of cumulative weighting require a later matched comparison. The current
V1 run remains a control; its raw training loss is not directly comparable to
this different weighted recipe.
