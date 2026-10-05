---
schema: ref2dex.probe.v2
probe_id: P-20261005-geometric-support
experiment_id: P-20261005-geometric-support
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: pending
claim_id: C3
hypothesis_family: HF-geometric-innovation
probe_index_in_family: 2
seed_pool: probe
seeds: [241, 242]
decision_changed_if_positive: eliminate multiplicative or coordinate scaling failure before considering spatial models
decision_changed_if_negative: stop PCA bilinear nominal endpoint route while preserving spatial and execution-model hypotheses
status: UNCLEAR
run_id: geometric-support-s241
---

# Separate physical point-flow from feature amplification

Result: UNCLEAR: fixed post-result diagnostic protocol before new fits.
Decision: Diagnose the source of ref10 error explosion, do not remove outliers or label original OI-CmV2 ineffective.

## Motivation and root Decision Note

Blocker for interpreting the ref10 representation screen: Flow I12.440 versus
State1.074; frozen shuffled action REDUCES error. Root GPU frozen-weight replay
matches all28 predictors,15Flow candidates,eight scorers EXACTLY; current FK
and labels valid. Top1/top5 factual rows produce57.84%/85.13% I error, while
Flow design test maximum251.74 versus train30.43. Independent reviewer found
all32 PCA scales above1mm floor: not division by zero, but genuine projected
support shift, amplified by unconstrained state×action products.

Question: is the collapse dominated by PCA score scaling, bilinear feature
extrapolation, or a poor physical representation even after removing both?
This changes whether to retain nominal point-flow features for spatial modeling
versus prioritize execution/contact response modeling. Cheapest discriminator:
same saved geometry/basis/outer split, six exact ridge fits on idleGPU6;
no new collection, no new hyperparameter tuning, no task transfer refit yet.
Source test has already been observed: explicitly post-result Decision Probe,
not independent confirmatory evidence. Do not modify original card's gates.

## Frozen factorial contract and stopping rules

Keep original CV E/current I baseline, H104+geometry16, α32, all854 rows and
train-only original normalizers/PCA. Same664slots including zero-padded
products. Factorial Flow action PCA score scale (original per-PC std versus
fixed20mm physical units) × bilinear products (512included versus zero).
Original whitened-products arm comes from already saved primary predictions;
refit the other three cells EXACTLY once. Add matched additive State/Arm/Joint
controls, six new fits total. No clipping/excluding errors, no α/component/
scale sweep, no selecting checkpoint by test. 20mm is a declared physical
coordinate convention, not a fitted optimum. All source local sample points,
geometry and assignments identical; no future state input.

Report E/I/joint, factual gains with2000 fixed-fit cluster bootstrap seed242,
all14same-H contrasts and arm-centered I. Geometry warrants another experiment
only if fixed-scale additive Flow I point gain≥3% over State/Arm/Joint AND
TrainMean, plus arm-centered contrast gain-vs-zero>0. Otherwise this specific
PCA/ridge endpoint route remains UNPROMISING; do not infer no action signal.
Report uncertainty and residual limitations rather than rescue by new gates.

Budget: GPU6 only, main timeout120s, outputs≤10MiB; no OOF/downstream/NN/sim.
Stop nonfinite/source drift/knownresource conflict, preserve failure artifacts.
Original source run79MiB, cumulative this round cap120MiB and≤840s GPU wall
including both main+audits; global Campaign still applies. Independent reviewer
must inspect anomalous-result attribution before closing this local route.

## Results, artifacts and limitations / future evidence

Pending. Source `outputs/cm-interaction-oracle/geometric-innovation-s241-r2/`;
output `outputs/cm-interaction-oracle/geometric-support-s241/`.
This factorial tests a fixed feature model, not the original spatial network.
Physical per-finger amplitudes unchanged; full tables retained in the source
[ref10 card](P-20261005-geometric-innovation.md). No formal claim, selector or
matched trained-policy utility. Independent datasets/seeds, execution-aware
surface input and spatial architecture remain future decision candidates.
