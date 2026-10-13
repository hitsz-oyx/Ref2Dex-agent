---
schema: ref2dex.probe.v2
probe_id: P-20261013-pointwam-repro-frozen-step
experiment_id: P-20261013-pointwam-repro-frozen-step
date: 2026-10-13
task: pointwam-repro
branch: main
git_commit: ea2575aa
claim_id: C1
hypothesis_family: HF-pointwam-repro-human-forecaster
probe_index_in_family: 4
seed_pool: debug
seeds: [12]
decision_changed_if_positive: keep this frozen-encoder training step; do not start 60k or global batch 192 from one clip
decision_changed_if_negative: do not train until the frozen-encoder backward fits on one 24 GB GPU
status: UNCLEAR
run_id: frozen-step-20261013-r1
---

# Frozen encoders, one optimizer step

Result: UNCLEAR. One AdamW step on the full qualified clip finished with finite loss 7.903 and gradient norm 5.592. The forecaster is still randomly initialized, and the horizon statistics come from this clip alone.
Decision: Do not start 60k or global batch 192. One step fits; that is not a reason to treat this clip as pre-training.

Class: Blocker. This asks whether Table 4's optimizer step runs when the scene and text features come from frozen Mosaic3D and Recap-CLIP. It does not ask whether the forecast matches future motion. `ea2575aa` is the parent commit; the task directory was still untracked.

## Protocol

- Clip: local copy of one official EgoDex TRAIN Part2 fit clip, all 5848 currently valid scene points, `training_allowed=false`. Language: “Disassemble a chair while sitting against a lavender background.”
- Scene encoder: frozen Mosaic3D `sc+ar+sc+++ma+st.ckpt`, condition `ScanNet`. Text encoder: frozen Recap-CLIP `ln_final` tokens. Both are removed from GPU memory before the forecaster step.
- Forecaster: randomly initialized `PointWAMForecaster`, 655,527,092 parameters. AdamW `1e-4`, weight decay `0.01`, gradient clip `5`, bfloat16 autocast, seed `12` from the debug pool.
- Horizon mean and standard deviation are fit on this clip only. Motion weights use the stored metric positions.
- GPU0 was idle. The remote GPU3 reconstruction was not touched.

## Result

14 text tokens, feature widths 768 and 1024, both finite. Loss 7.903 and gradient norm 5.592, both finite. The reported norm is the value before clipping. Peak allocated memory 13028 MiB. The optimizer step took 0.604 s; the whole process, including both checkpoint loads, took 22.543 s. Recorded in
[result.json](../../../../../../outputs/pointwam-repro/frozen-step-20261013-r1/result.json).

## Decision and limits

A finite step shows the frozen features can supervise the paper-sized module for one update. It is not evidence that the reimplementation matches PointWAM. Official training code and checkpoints remain unpublished, and the 48-clip subset stays `training_allowed=false`.
