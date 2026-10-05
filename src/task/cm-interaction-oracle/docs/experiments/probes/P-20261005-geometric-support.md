---
schema: ref2dex.probe.v2
probe_id: P-20261005-geometric-support
experiment_id: P-20261005-geometric-support
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: 5a50f1e
claim_id: C3
hypothesis_family: HF-geometric-innovation
probe_index_in_family: 2
seed_pool: probe
seeds: [241, 242]
decision_changed_if_positive: eliminate multiplicative or coordinate scaling failure before considering spatial models
decision_changed_if_negative: stop PCA bilinear nominal endpoint route while preserving spatial and execution-model hypotheses
status: UNPROMISING
run_id: geometric-support-s241
---

# Separate physical point-flow from feature amplification

Result: UNPROMISING: additive physical-scaled flow avoids the bilinear explosion but still underperforms State, Arm and Joint; the isolated original geometry hypothesis remains open.
Decision: Stop PCA/ridge endpoint fitting; retain geometry contract and require local spatial or execution inductive bias for the next geometry experiment.

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

Source `outputs/cm-interaction-oracle/geometric-innovation-s241-r2/`;
output `outputs/cm-interaction-oracle/geometric-support-s241/`.
Actual code5a50f1e, GPU6 sixfits0.97s, peaksee manifest/result;1.9MiB artifacts.
Identical fixed original688/166split and physical data, no outlier exclusion.

| Scaler / representation | State-action products | E MSE | I MSE |
| --- | --- | --- | --- |
| Flow per-PC std (saved original) | yes | 13.90584 | 12.44019 |
| Flow per-PC std | no | 1.82714 | 1.25431 |
| Flow fixed20mm | yes | 3.33273 | 2.08974 |
| Flow fixed20mm | no | 1.70895 | 1.13840 |
| State | no | 1.60434 | 1.07388 |
| Arm | no | 1.63293 | 1.05502 |
| Joint | no | 1.60924 | 1.04144 |

Removing products reduces I error89.92%(95%57.34–94.20); physical scale
with products reduces83.20%(45.06–88.16). Holding additive family fixed,
physical scale improves over per-PC standardization9.24%(3.28–14.59).
These diagnose sensitivity to extrapolation and implicit feature regularization;
20mm is not identified as an optimal scale. The physical additive arm still
worsens I over State6.01%(gainCI−11.79..−.23),Joint9.31%(−16.75..−2.52),
Arm7.90%(−14.75..−1.64),TrainMean27.81%(−81.06..5.27).
Flow physical additive Icontrast rawcorr.23668,sign56.57%,gainzero4.82%;
centeredcorr.27209/sign63.70%/gainzero6.62%. Joint additive centeredgain7.41%.
The geometry superiority screen is false even after reducing amplification.

Root attribution: unconstrained multiplicative features cause most of the
original collapse; PCA tail scaling contributes. Neither numeric replay nor
additive stabilization identifies a meaningful local action-consequence model.
Do not call the original physical representation refuted or interpret a
gain against the collapsed initial arm as research success. Keep allfailed
and successful runs; independent review must incorporate this factorial.
This factorial tests a fixed feature model, not the original spatial network.
Physical per-finger amplitudes unchanged; full tables retained in the source
[ref10 card](P-20261005-geometric-innovation.md). No formal claim, selector or
matched trained-policy utility. Independent datasets/seeds, execution-aware
surface input and spatial architecture remain future decision candidates.
