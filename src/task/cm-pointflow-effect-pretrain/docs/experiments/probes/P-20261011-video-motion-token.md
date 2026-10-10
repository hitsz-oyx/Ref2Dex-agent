---
schema: ref2dex.probe.v2
probe_id: P-20261011-video-motion-token
experiment_id: P-20261011-video-motion-token
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: pending
claim_id: C1
hypothesis_family: HF-video-point-dynamics
probe_index_in_family: 1
seed_pool: probe
seeds: [226]
decision_changed_if_positive: retain a DynamicWAM-style global history token for the weak-video point-flow auxiliary and test representation transfer
decision_changed_if_negative: defer global motion-token fusion and prioritize calibrated wrench/simulator routes
status: PLANNED
run_id: video-motion-token-20261011-r1
---

# Metric-3D history motion-token Probe

Class: Decision. This Probe tests whether a shared interval summary improves the
existing weak-video point-flow learner. It preserves the current per-point
history features and correspondence-aware Huber flow loss; the new token is an
auxiliary conditioning path only.

## Decision note

**Question:** does a global history motion token help beyond the current local
velocity/acceleration/displacement features on the qualified H8 video view?

**Evidence:** DynamicWAM exposes interval motion and exact timing through a
separate token path, while the current `VideoPointWorldWM` has only point-wise
history features. The public DynamicWAM 12D contract is image-plane; this Probe
uses an explicit metric-3D adaptation rather than silently treating 2-D values
as world motion. ParticleFormer's unordered geometry loss is deferred because
these weak tracks already carry identity and the native rigid route needs it.

**Cheapest test:** derive three history intervals from H4 using only observed
frames and timestamps. For object/background groups compute 15 metric-3D
features (mean displacement xyz, mean/p99 displacement, dt, mean velocity xyz,
mean/p99 speed, mean acceleration xyz, mean acceleration magnitude), normalize
with train-only statistics, and feed them through a small MLP plus interval and
group embeddings. Compare a matched control with the same initialized token
module disabled against the enabled arm, using the existing H8 static/CV gate.

**Gate:** a positive Probe requires the enabled arm's object H8 clip macro to be
at least 10% better than the matched control, no qualifying clip over 20% worse,
and background H8 no worse than 1.2 times static. A negative result closes only
this global-token recipe, not all video, point-dynamics, or wrench routes.

## Fixed protocol

- Source: existing `video-point-dynamics-20261010-r1/data-h8-view-r5` pack;
  every sequence hash is rechecked before training.
- Split: source video IDs are disjoint between train and dev; no future frame
  participates in point selection, token statistics, or normalization.
- Model: same PTv3-small, point feature contract, Huber flow loss, and H8
  evaluator as the previous video Probe. Only the token path differs.
- Match: control and motion arms use the same seed 226, same initial full model
  state including the token module, and the same random window sequence.
- Compute: one idle GPU, 500 updates per arm, at most 15 minutes and 5 GiB
  output. GPU utilization and memory are recorded at progress checkpoints.

## Results

Pending execution.

## Limitations / future evidence

The source point tracks are offline teacher geometry from ObjectForesight and do
not establish a causal deployment camera pipeline. The weak-video corpus has
few qualifying object points in dev, so the result is a route Probe only. A
positive token result would still require native encoder-transfer and matched
Cm-on/Cm-off policy validation. A negative token result does not reject RBO,
T-Rex, CHORD simulator support, or correspondence-free geometry losses on an
explicitly unordered label contract.
