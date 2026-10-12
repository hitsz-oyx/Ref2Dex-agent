---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-compact-forecast
experiment_id: P-20261012-goal-point-wam-compact-forecast
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 11
seed_pool: probe
seeds: [247, 248, 249]
decision_changed_if_positive: use explicit observed geometry as a learned forecasting substrate and test what WAM adds under matched inputs
decision_changed_if_negative: stop this conditional forecast screen and audit supervision/budget before more architectural changes
status: UNCLEAR
run_id: compact-forecast-20261012-r1
---

# Compact real-input forecasting diagnostic

Class: Decision. Trajectory representation supports the final self-trained RL/Cm
claim. Linear history improves only OakInk2; the conditional full query trial
is stopped. Specific question: can a small directly observed geometry learner
beat zero hand displacement at the same sample/update budget, and does object
goal add measurable information? Cheapest test omits expensive scene/world
computation and caches only compact features/labels, without changing task goals.

Result: Pending.
Decision: Run matched H-only/H+object-goal compact models, then stop this bounded turn.

## Protocol fixed before execution

- Exact frozen train corpus/stats,1200 precomputed mixed_indices seed248 and
  the existing96 val indices seed249. No added training samples or extra epochs:
 600 updates, microbatch2, one pass in the original stream. Cache all compact
  tensors and indices with hashes. Future hand is label only, no oracle hand goal.
- H input: current hand XYZ/.05m, observed9D dynamics normalized with existing
  scene stats3:12, current target-object centroid/coordinate standard deviation
  in .05m units. Per hand/keypoint/time learned8D IDs and shared51->128->128->3
  MLP predict relative trajectory points. No source/domain ID input.
- H+goal adds9D future target geometry: centroid displacement/.05m and first
  two columns of (R-I), where R is Kabsch from existing paired object start/goal
  points. H-only zeros these9D entries and keeps the exact same observed object
  geometry. No executed future hand or future non-target object features enter.
- Identical initial parameter hashes, zero final layer, seed247. BF16 GPU2,
  AdamW lr.001/decay.01,50-update warmup/cosine to10%, grad cap1. No world loss,
  pretrained weights or synthetic labels; compact diagnostic checkpoints cannot
  initialize WAM/policy training. Record train and val EPE, goal ablation, history
  each100 updates. Decision uses FINAL600 only.
- History positive: H-only EPE >=5% below zero on >=2 sources. Goal positive:
  H+goal EPE >=5% below H-only and zero and its own static-goal ablation on
  >=2 sources. Report these gates separately: history success alone is not
  goal-conditioning evidence. Otherwise UNCLEAR, no architecture refutation.
- This is NOT a matched architectural comparison against WAM: inputs/shape,
  scene omission and world supervision differ. Same data/update budget and two
  matched compact input arms permit a cheap learnability screen only. Attribution
  to PTv3/world loss requires a subsequent matched intervention.
- Shared current-turn cap25GPUmin/40min wall/6GiB new artifacts. This diagnostic
  <=480s wall/GPU2 and <=64MiB artifacts, stop on nonfinite/OOM/identity drift/
  foreign process/deadline. CPU data packing; all neural learning/evaluation GPU.

## Decision Note

WAM outputs remain near static, a controlled teacher can induce goal response,
and simple history only helps one source. Root chooses a compact actual-data
learner to test available information before any further full architecture or
training scale changes. Positive narrows the representation/learning question;
negative returns to supervision or budget diagnosis. No mission/claim or external
boundaries change, no new branch or extra authorization required.

## Limitations / future evidence

Single seed and96 validation windows are exploratory. Compact features discard
scene contacts/other objects and object shape detail; failure cannot refute the
full-input method. Goal success is conditional real-prediction evidence, not
physical hand feasibility, world utility, stable grasping or RL efficiency.

## Artifacts

- [Inputs and identities](../../../../../../outputs/goal-point-wam/compact-forecast-20261012-r1/input_manifest.json)
- [Results and separate gates](../../../../../../outputs/goal-point-wam/compact-forecast-20261012-r1/result.json)
- [Implementation](../../../tools/audit/probe_compact_forecast.py)
