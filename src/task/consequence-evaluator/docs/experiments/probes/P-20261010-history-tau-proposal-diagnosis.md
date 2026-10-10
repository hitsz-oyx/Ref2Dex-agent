---
schema: ref2dex.probe.v2
probe_id: P-20261010-history-tau-proposal-diagnosis
experiment_id: P-20261010-history-tau-proposal-diagnosis
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: e45fa8e
claim_id: C3
hypothesis_family: HF-history-tau-proposal
probe_index_in_family: 1
seed_pool: probe
seeds: [296]
decision_changed_if_positive: repair history representation before investing in multimodal tau proposal
decision_changed_if_negative: prioritize train-only candidate coverage and multimodal proposal over more deterministic regression
status: UNCLEAR
run_id: history-tau-proposal-diagnosis-20261010-r1
---

# Diagnose the history-to-candidate-tau blocker

## Motivation / Decision Note

User explicitly redirects research through [ref8](../../user/ref/ref8.md): focus
on usable H-to-candidate-tau, freeze PointWorld, and do not repeat GT E/I-to-Y
necessity experiments. C3/Mission are unchanged. Existing observed tau-only
ranking .7007 and shuffle .4599 are approximate-H evidence from seven informative
anchors; they do not pass a formal same-state ranking Validation. The previous
single-tau MLP improves train/val but held seed414 RMSE .30280m exceeds current
hand persistence .26214m. Historical split seeds412/413/414 have already been
explored and are reused as Probe data, not fresh Validation holdout.

Classify this as a Blocker/Decision audit: distinguish train-only input scaling,
frame/split drift, and one-to-many future ambiguity before paying for a larger
proposal model. Cheapest first step replays frozen weights/data. Fixed saved
prediction replay reproduces .30279982 versus .26213902m (6144/4096/4096 windows).

Ranked hypotheses: near-constant training channels amplify unseen variation;
coordinate or source-distribution mismatch; deterministic prediction averages
multiple possible structured-controller futures. Hypotheses are not conclusions.

## First fixed protocol

Read/hash the previous best checkpoint and its original three source manifests/
trajectories. Reuse exact collection, frame transform and train-only statistics.
GPU2 inference must reproduce saved test predictions. Report per-split normalized
input extrema, train-floor channels, target/current displacement, and per-tick
errors. Two fixed **diagnostic** interventions, no fitting or test-based choice:
zero train-floor normalized channels (training std<=1.01e-4), or clamp normalized
inputs to [-10,10]. Apply each consistently to every split. Neither changes the
frozen checkpoint or task target. Repaired deployable preprocessing would require
a separate matched fit; frozen-weight interventions are only diagnostic.

Stop on hash/source/schema/split/shape/replay mismatch or nonfinite. NN inference
uses an idle GPU, statistics/file transforms CPU. No oracle future object, future
q, force or U32 label enters proposal inputs; old H is retained verbatim and its
semantic content must be audited before calling it deployable sensor history.

If a specific preprocessing defect is found, permit one bounded matched repair
from the same initialization/1200steps, train-only statistics and val selection,
seed296. If it is not found, the cheapest next discriminator is train-only
nearest-neighbor candidate displacement coverage before CVAE/diffusion training.
Specify that follow-up before running. Best-of-K against GT is only coverage,
never a deployable selector or guaranteed execution value.

## Resources / boundaries

One idle GPU, total<=16GPUmin/1GiB; each replay<=180s, each fit<=600s. Fresh output
directories, no overwrites/new branch/push. Shared300GB/four-GPU global bounds
still apply. Monitor utilization/memory/ETA and preserve foreign processes.
No new simulation, τ→A retraining, E/I necessity fit, PointWorld/online selector,
or formal claim. Literature compares ACT/CVAE/DP/PointWAM/DexWM against actual
input/target contracts, not model names alone.

## Results

Pending. Artifacts: outputs/consequence-evaluator/history-tau-proposal-diagnosis-20261010-r1/.

## Limitations / future evidence

Single motion and structured actor plus random residual futures; action plans
are not supplied to H-only model. Split separation does not alone guarantee
matched distributions or actual sensor-only history. Strong candidate execution,
exact-state ranking, held-out-motion generalization and Cm-on/off remain deferred.
