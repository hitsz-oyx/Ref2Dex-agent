---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-hand-query
experiment_id: P-20261012-goal-point-wam-hand-query
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: dfcfef3a185c82ea4494799316f8c3d293857e0a
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 10
seed_pool: probe
seeds: [247, 248, 249]
decision_changed_if_positive: retain explicit current-hand query conditioning and investigate conditional goal use before robot integration
decision_changed_if_negative: stop this short-budget hand-query route and inspect supervision or joint-loss learning without claiming architectural refutation
status: UNCLEAR
run_id: hand-query-20261012
---

# Current-hand history and direct query conditioning Probe

Class: Decision. Supports the trajectory representation subgoal preceding the
final self-trained RL/Cm comparison. Latest real-data heads predict near zero
and do not use goal; units/paired-goal tuning did not help. Distinguish missing
accessible hand dynamics from broader learning/supervision limitations. Cheapest
screen is observed-only velocity forecasting on the existing96 validation
windows, with independent raw history reconstruction before any neural change.

Result: UNCLEAR; observed velocity forecasting helps OakInk2 but fails the multi-source gate.
Decision: Stop the conditional full-WAM query training here; use a separately registered compact predictor to test input learnability.

## Phase1: observed-only statistics fixed before execution

- Same frozen main-three corpus and exact96 val indices from real-goal-scale
  raw run. No test panel, sample selection, learned model or goal/future-hand
  input into the predictions. Future hand is only the evaluation target.
- Recover current hand by semantic IDs and observed features3:12. Velocity is
  last-step delta*30; acceleration delta2*900; history displacement spans3steps.
  Independently reconstruct all from raw t-3..t positions in the single current
  anchor frame, check at2e-4 tolerance (includes acceleration units).
- Predict zero displacement, last velocity*t/30, and primary three-step average
  displacement*t/3, for t1..24. Same valid hand points and trajectory EPE as
  the preceding Probe. Primary positive screen: >=5% EPE improvement over zero
  on >=2 sources. All per-source and per-window results retained.
- CPU permitted for pure data/label statistics, no neural inference. If primary
  gate fails, stop hand-history query training here; reassess supervision before
  further training. If it passes, test the explicit query below.

## Phase2: conditional matched neural Probe

- Two fresh seed247 PW-initialized endpoint-goal WAMs, goal units .05m. Both
  instantiate the identical additional12-to128-to128 geometry MLP. Only routing
  switch differs: disabled control versus enabled current hand XYZ+observed
  velocity/acceleration/history displacement added to each hand/keypoint query.
  Identical full initial parameter hashes and prediction before zero-initialized
  query residual learns. No extra goal supervision or output motion prior.
- Current position units .05m;9 observed dynamics use existing scene stats3:12.
  Apply current validity masks before normalization/MLP, zero invalid residuals.
  Query still retains learned time/hand/keypoint IDs; world never receives new
  hand query or goal. Historical default configurations keep original state keys.
- Same train stream1200windows seed248, exact96 val seed249, losses and600
  updates: microbatch2, world_weight1, BF16, AdamW lr.001/decay.01,50 warmup,
  cosine to10%, grad cap1. Final600 determines comparison, no best selection.
- Four-update CUDA smoke first, gradient/reload/world isolation checks. Frozen
  audit evaluates same true/static-goal ablation and zero/geometric baselines.
- Hand-query PROMISING requires enabled EPE >=5% lower than disabled AND zero
  displacement on >=2 sources, with moving-world regression <=5% on all
  sources. Report goal-use separately; success without goal use is a history
  prediction signal, not goal-conditioned behavior or robot/RL utility.
- One idle GPU2, <=25GPUmin/40min wall for this turn, <=6GiB new artifacts against
  current263GB/300GB cap. Stop on nonfinite, OOM, identity mismatch, foreign
  process or600s training/arm. No new branch/download/robot/RL or external writes.

## Decision Note

Several short runs remain near stationary. Root chooses a cheap real observed
history screen and conditional direct-query comparison, rather than extending
units/epochs or assuming a disconnected goal path. Positive routes to explicit
hand representation investigation; negative stops this local query trial and
moves to supervision/joint-loss discrimination. This decision keeps the mission,
claim and global boundaries; no extra authorization is needed. User confirms
there may be no new modifications and authorizes continued autonomous exploration.

## Limitations / future evidence

One seed and96 familiar val windows cannot validate scientific or robot utility.
A linear history screen is only an information/cheap-baseline test. Query mode
can help without useful goal use; goal dependence needs its own positive evidence.
World-loss-off, longer training and multiple seeds remain conditional future work.

## Artifacts

- [Observed-history screen](../../../../../../outputs/goal-point-wam/hand-history-20261012-r1/result.json)
- [History screen source](../../../tools/audit/probe_hand_history.py)

## Completed Phase1 / conditional Phase2 not launched

Run code dfcfef3a, label-statistics runtime1.00s on CPU. Raw history reconstruction
maximum error0 in all sources; semantic/missing/current-only tests pass.

| Source | Zero / last velocity / three-step velocity EPE (mm) | Primary improvement |
| --- | ---: | ---: |
| OakInk2 | 34.80 / 27.68 / 30.71 | +11.73% |
| GRAB | 141.32 / 148.77 / 156.57 | -10.79% |
| ARCTIC | 71.63 / 91.25 / 94.76 | -32.29% |

Only OakInk2 passes; primary >=2-source gate fails. Its last-velocity secondary
reference improves20.45%, but other sources worsen. Do not select this secondary
post hoc as a passed multi-source screen. The planned full-WAM query change and
training are NOT implemented/launched. This does not show that nonlinear hand
history is uninformative or that explicit hand position cannot help.

Decision Note: retain the negative linear-baseline results and stop the full
query trial specified by this card. A compact learned H-only/H+object-goal
predictor can distinguish cheap input learnability from failure of this rigid
linear forecast before investing in another full scene/world training pair.
It is a separate Probe with no architecture-causality claim.
