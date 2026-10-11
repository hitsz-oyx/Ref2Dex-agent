---
schema: ref2dex.probe.v2
probe_id: P-20261011-egotouch-processed-history
experiment_id: P-20261011-egotouch-processed-history
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 5549eb1
claim_id: C1
hypothesis_family: HF-egotouch-pressure-contact
probe_index_in_family: 2
seed_pool: probe
seeds: [229]
decision_changed_if_positive: open a processed hand-surface pressure/contact token before native point-cloud fusion
decision_changed_if_negative: defer EgoTouch pressure and prioritize T-Rex/RBO/simulator wrench routes
status: UNPROMISING
run_id: egotouch-processed-pressure-history-20261011-r3
---

# Causal processed-pressure history probe

Class: Decision. This follow-up isolates the part that the first pressure card
did not test: whether a past processed `(21,21)` pressure map contains temporal
information beyond persistence. It does not claim calibrated force, object-world
contact, CHORD supervision, or downstream Cm utility.

## Decision note

**Question:** does a processed pressure/contact history provide a useful
hand-surface auxiliary signal once the released record-wide normalization is
removed from the causal input?

**Evidence:** the first local Probe consumed raw 512-byte channels and lost to
persistence on all five held tasks. The source pack has finite, masked processed
grids for the same ten official TRAIN records, so that result does not test the
processed-history hypothesis.

**Cheapest test:** use the existing immutable pressure pack, side-specific H4/K8
windows, one seed, 500 GPU updates, and matched persistence. For every window,
compute one scale from the maximum measured value in the past four frames only;
apply that scale to both future targets and persistence. Train masked pressure
Huber plus relative activation BCE, with and without past hand shape.

**Stop/gate:** stop on source hash drift, invalid masks, nonfinite loss, or GPU
contention. A positive signal requires `pressure_hand` to beat persistence by at
least 10% on at least three held tasks with no held task more than 20% worse. A
negative signal closes only this causal processed-history recipe and leaves the
calibrated T-Rex/RBO and simulator CHORD routes open.

## Fixed protocol

- Source pack: `outputs/cm-pointflow-effect-pretrain/egotouch-pressure-targets-20261011-r2`;
  its manifest and per-record hashes are loaded before training.
- Split: five fit records and five held tasks from the previously frozen pack;
  no record is moved after looking at future arrays.
- Inputs: past processed pressure values, their finite-cell masks, hand-local
  shape/validity in the optional arm, and side identity. No raw glove bytes or
  future hand frames are input.
- Scaling: `scale=max(max(past measured pressure), 0.05)`, computed separately
  for each window. The model sees only pressure divided by this past scale;
  target and persistence use the same scale. The future record-wide maximum is
  therefore not an input statistic.
- Target: masked future pressure for K=8 and a relative activation proxy
  `valid & (pressure/scale >= 0.05)`. This proxy is not physical contact GT.
- Compute: seed 229, batch 32, AdamW, 500 updates, one currently idle GPU;
  output is a new run directory and does not overwrite prior checkpoints.

## Results

The reproducible run used all 10 records (3,166 fit windows and 3,924 held
windows), seed 229, batch 32, AdamW, 500 updates, and GPU0. Both arms completed
normally; elapsed time was 9.6 seconds for pressure-only and 8.9 seconds for
pressure+hand. The result manifest records the committed implementation
`5549eb1` and pack hash
`315a32ef58d3ebe660c40638602988a3ea6bd741cd697326f36b8343504941cf`.

Original-scale held pressure MAE / persistence MAE was:

| held task | pressure history | pressure history + hand | persistence |
|---|---:|---:|---:|
| drag chair | 0.0488 / 0.0299 | 0.0467 / 0.0299 | 0.0299 |
| flip book pages | 0.0706 / 0.0330 | 0.0738 / 0.0330 | 0.0330 |
| badminton racket | 0.0613 / 0.0182 | 0.0622 / 0.0182 | 0.0182 |
| shop for snacks | 0.0398 / 0.0180 | 0.0397 / 0.0180 | 0.0180 |
| squeeze pliers | 0.0288 / 0.0116 | 0.0286 / 0.0116 | 0.0116 |

Both causal processed-pressure arms lost to persistence on every held task. This
Probe is therefore `UNPROMISING` for direct future pressure-map forecasting,
even after removing the record-wide scale from the input. It does not test a
learned contact token as a downstream policy feature, T-Rex per-fingertip
wrenches, RBO measured interaction wrenches, or simulator CHORD support.

## Limitations / future evidence

The source grid remains a normalized tactile signal, not Newtons. The pressure
map has sparse mapped cells and no verified object-world calibration. Even a
positive held-task result only justifies a hand-surface representation Probe;
native fusion still requires a matched Cm-on/Cm-off evaluation. T-Rex finger
wrenches, RBO measured interaction wrenches, and CHORD simulator support remain
separate routes.
