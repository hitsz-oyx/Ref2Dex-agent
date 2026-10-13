---
schema: ref2dex.probe.v2
probe_id: P-20261013-pointwam-repro-volt-forward
experiment_id: P-20261013-pointwam-repro-volt-forward
date: 2026-10-13
task: pointwam-repro
branch: main
git_commit: ea2575aa
claim_id: C1
hypothesis_family: HF-pointwam-repro-human-forecaster
probe_index_in_family: 2
seed_pool: debug
seeds: [11]
decision_changed_if_positive: keep the paper-sized Volt-H and load frozen encoders before any 60k run
decision_changed_if_negative: do not shrink the backbone silently; report that one 24 GB GPU cannot hold Table 4
status: UNCLEAR
run_id: volt-h-forward-20261013-r1
---

# Paper-sized Volt-H, one synthetic step

Result: UNCLEAR. One bf16 step on 12000 random points finished in 1.224 s at 12776 MiB, but the features are not Mosaic3D or Recap-CLIP.
Decision: Do not start 60k or global batch 192. Load the frozen encoders before any clip training.

Class: Blocker. This asks whether the published Volt-H forecaster can take one
optimizer step on a 24 GB GPU without shrinking width or depth. It does not
ask whether the model learns human motion. `ea2575aa` is the parent commit;
the task directory was still untracked.

## Protocol

- Model: 1 cm sparse stem, 5³ patches, 32 layers, width 1280, 20 heads,
  axial RoPE on vision tokens, zero-initialized fusion, separate heads.
- Trainable parameters: 655,527,092. The paper lists 651M for pre-training.
- Optimizer: AdamW `1e-4`, weight decay `0.01`, gradient clip `5`, constant
  schedule, bfloat16 autocast. Seed `11` from the debug pool.
- Input: 12000 synthetic points inside a 0.8 m cube, 20 hand keypoints, 32
  text tokens. Scene width 768 and text width 1024 match the frozen-encoder
  contract. Values are random.
- GPU0 was idle. The remote GPU3 reconstruction was not touched.

## Result

Loss 2.073 and gradient norm 1.883, both finite. Peak allocated memory
12776 MiB. Total process time 6.716 s, of which the step was 1.224 s.
Recorded in
[result.json](../../../../../../outputs/pointwam-repro/volt-h-forward-20261013-r1/result.json).

## Decision and limits

A finite step only shows that the trainable module fits. It is not evidence
that the reimplementation matches PointWAM, and it is not a reason to train
for 60k steps on random features or on the 48-clip subset. At this step time,
accumulating a global batch of 192 on one GPU would take on the order of
months. Official code and checkpoints remain unpublished.
