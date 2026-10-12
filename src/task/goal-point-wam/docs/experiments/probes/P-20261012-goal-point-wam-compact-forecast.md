---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-compact-forecast
experiment_id: P-20261012-goal-point-wam-compact-forecast
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: b24dc66a869bcc86cdc865fd188f6271edf679ac
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 11
seed_pool: probe
seeds: [247, 248, 249]
decision_changed_if_positive: use explicit observed geometry as a learned forecasting substrate and test what WAM adds under matched inputs
decision_changed_if_negative: stop this conditional forecast screen and audit supervision/budget before more architectural changes
status: PROMISING
run_id: compact-forecast-20261012-r1
---

# Compact real-input forecasting diagnostic

Class: Decision. Trajectory representation supports the final self-trained RL/Cm
claim. Linear history improves only OakInk2; the conditional full query trial
is stopped. Specific question: can a small directly observed geometry learner
beat zero hand displacement at the same sample/update budget, and does object
goal add measurable information? Cheapest test omits expensive scene/world
computation and caches only compact features/labels, without changing task goals.

Result: PROMISING for compact observed-input prediction; object-goal incremental utility remains UNCLEAR.
Decision: Retain the compact forecasting diagnostic; next use matched WAM hand-query routing to localize the representation gap, with world-loss attribution deferred.

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

## Completed results / hypothesis distinction

Run code b24dc66a. Both600-update arms completed; input/data/stats identities
match, exact1200 train and96 validation indices cached. Identical initial full
parameter SHA256 b348cd1553be0a18e5f4e181986904b39cba905ee848e2444478b2fc34706bd2.
Each model has23,851 parameters, no source ID or future hand input, no world
loss, no pretrained weights, no synthetic teacher labels. The zero-initialized
output has identical initial predictions in both input arms. Known rigid goal
geometry, actual conditional gradient, future-hand independence and semantic
current-history tests pass (13 tests passed,1 optional CUDA test skipped).

| Source | Zero / H-only / H+goal val EPE (mm) | H-only vs zero | Goal vs H-only |
| --- | ---: | ---: | ---: |
| OakInk2 | 34.80 / 26.36 / 26.73 | +24.24% | -1.38% |
| GRAB | 141.32 / 130.72 / 128.38 | +7.50% | +1.79% |
| ARCTIC | 71.63 / 64.90 / 62.92 | +9.40% | +3.05% |

The primary observed-input learning gate passes all3 sources. Overall local
**PROMISING for compact observed-input forecasting**, not for goal utility or
any robot/RL claim. H-only train EPE27.18/126.75/69.13mm also beats zero by
17.91%/12.02%/8.87%; no train/val accuracy collapse is evident on this screen.
The same data/sample/update budget contains a cheaply learnable signal despite
poor constant-velocity forecasting on two sources and near-zero WAM prediction.

Object-goal incremental gate fails all3 sources. Goal versus its own static
geometry ablation gives -1.27%/+1.38%/+1.43% EPE improvement, below5%. Unlike
the WAM replay, the compact model does react: mean trajectory response to goal
removal is2.15/13.17/7.35mm. Response magnitude alone is not useful conditional
accuracy. Keep the goal question **UNCLEAR**, do not interpret the history
positive as proof that replacing language with goal succeeds or fails.

## Final Decision Note and resource closure

Observed geometry/dynamics are learnable by this cheap model in the same1200
sample/600-update screen. Root retains it as a diagnostic substrate and selects
the next minimum matched question: same WAM modules/initialization/losses/data
with direct observed hand XYZ/history query residual enabled versus disabled.
That would isolate query access; only afterward compare world loss. Current
results change representation-investigation priority without changing the core
mission/claim or attributing the gap exclusively to PTv3 or world supervision.
The original linear-screen negative and skipped full-query arm remain intact.
No further training starts after this bounded diagnostic sequence.

GPU2 manifest/runtime samples record373MiB peak, utilization1--10% H-only and
6--19% H+goal. Total packing/training/evaluation17.38s; learning+final audits
4.37/3.88s. All600 updates/arm completed, all tensors/gradients finite, GPU2
released. Artifacts about10.4MiB, far below64MiB diagnostic and6GiB turn caps.
Small CPU tests/statistics avoid GPU startup; neural training/evaluation use GPU.
The short run completed before the first interactive GPU status read, so resource
assessment uses recorded in-run samples rather than a fabricated live estimate.

## Additional limitations

The96 val windows have been inspected in prior Probes; they are an exploration
panel, not untouched Validation data. Explicit hand geometry, condensed object
features, lower model capacity and omission of scene/world all differ from WAM.
The two compact arms are matched to each other, not to historical WAM. A causal
architecture/loss claim needs the planned matched interventions and formal
Validation before any major route or final claim change.

- [H-only detailed metrics and train/val histories](../../../../../../outputs/goal-point-wam/compact-forecast-20261012-r1/h_only/result.json)
- [H+goal detailed metrics and goal ablation](../../../../../../outputs/goal-point-wam/compact-forecast-20261012-r1/h_goal/result.json)
- [Compact real tensors and labels](../../../../../../outputs/goal-point-wam/compact-forecast-20261012-r1/compact_data.pt)
