---
schema: ref2dex.probe.v2
probe_id: P-20261013-pointwam-repro-frozen-forward
experiment_id: P-20261013-pointwam-repro-frozen-forward
date: 2026-10-13
task: pointwam-repro
branch: main
git_commit: ea2575aa
claim_id: C1
hypothesis_family: HF-pointwam-repro-human-forecaster
probe_index_in_family: 3
seed_pool: debug
decision_changed_if_positive: keep this frozen-encoder wiring and only then consider a real clip step
decision_changed_if_negative: do not train on random features; fix the Mosaic3D or Recap-CLIP load path first
status: UNCLEAR
run_id: frozen-forward-20261013-r1
---

# Frozen encoders, one small forward

Result: UNCLEAR. Frozen Mosaic3D and Recap-CLIP produced finite Volt-H displacements on 1024 points of one qualified clip. The forecaster weights are still random, so this is an interface check, not a motion result.
Decision: Do not start 60k or global batch 192. The encoder path can be reused for a later training step.

Class: Blocker. This asks whether the public Mosaic3D checkpoint and Recap-CLIP can feed the paper-sized forecaster on an existing EgoDex clip. It does not ask whether the forecast matches the future motion. `ea2575aa` is the parent commit; the task directory was still untracked. The smoke script does not set a torch seed.

## Protocol

- Clip: local copy of one official EgoDex TRAIN Part2 fit clip, `training_allowed=false`. Language: “Disassemble a chair while sitting against a lavender background.” Scene points capped at 1024.
- Scene encoder: frozen Mosaic3D `sc+ar+sc+++ma+st.ckpt`, condition `ScanNet`, 2 cm grid, color `rgb/127.5-1`. Checkpoint keys match the local PPT/SpUNet exactly.
- Text encoder: frozen Recap-CLIP `ln_final` tokens, width 1024.
- Forecaster: randomly initialized `PointWAMForecaster`, bf16 autocast, `no_grad`. GPU0 was idle. The remote GPU3 reconstruction was not touched.

## Result

1024 scene points, 14 text tokens, feature widths 768 and 1024. Scene and hand displacements are both finite. Absolute means are 0.110 and 0.116 in standardized space. Peak allocated memory 6479 MiB. Wall time 22.572 s, including both checkpoint loads. Recorded in
[result.json](../../../../../../outputs/pointwam-repro/frozen-forward-20261013-r1/result.json).

## Decision and limits

A finite no-grad forward shows the frozen features enter the paper-sized module. It is not evidence that the reimplementation matches PointWAM, and the random trajectory head is not a pretrained forecast. Official training code and checkpoints remain unpublished. The 48-clip subset stays `training_allowed=false`.
