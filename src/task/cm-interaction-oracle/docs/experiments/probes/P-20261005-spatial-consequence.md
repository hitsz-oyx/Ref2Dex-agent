---
schema: ref2dex.probe.v2
probe_id: P-20261005-spatial-consequence
experiment_id: P-20261005-spatial-consequence
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: pending
claim_id: C3
hypothesis_family: HF-spatial-consequence
probe_index_in_family: 1
seed_pool: probe
seeds: [245, 246]
decision_changed_if_positive: prioritize nominal spatial candidate mechanism with an independent frozen evaluation
decision_changed_if_negative: separate local spatial prediction from nominal execution limitations before collecting again
status: UNCLEAR
run_id: spatial-consequence-s245
---

# Does shared local spatial structure preserve intended action consequences?

Result: UNCLEAR: fixed protocol, execution pending.
Decision: Test reduced OI-CmV2 spatial encoding on existing randomized windows before another simulation or policy run.

## Motivation and root Decision Note

Continue the user's ref10 physical geometric-action direction. The prior PCA/
ridge screen introduced unconstrained products and failed extrapolation;
its additive variant still underperformed. That evidence did not test the
OI-CmV2 local spatial encoder. This Decision serves Mission objective B:
does an action-conditioned physical representation supply extra information
for eventual trained-policy utility? Cheapest discriminator reuses all854
ref7 windows and ref8 train688/test166 environment split, no new simulation.

Root chooses a single reduced from-scratch V13 spatial Probe, persistence
innovation targets and matched direct task controls. Alternative acquisition
of realized execution histories costs simulation and changes data; postpone
until this clean spatial test identifies the remaining bottleneck. No seed,
width, learning-rate or checkpoint sweep; no test-selected stopping. Test
split has already been exposed in prior Probes: exploratory, not Validation.
Negative stops this fixed fit, not Cm/geometry. Positive requires independent
candidate mechanism evidence before policy training. No new authorization.

Single idle GPU6, wall cap1200s main +120s engineering smoke, outputs≤200MiB.
Stop on nonfinite, source/sample/hash drift, FK error>0.1mm, leakage, or resource
conflict; preserve failures. CPU synthetic integration tests only, because
GPU startup dominates; all real FK/PCA/model training/inference uses GPU.
Statistical/bootstrap work uses CPU. Independent review required on anomalies.

## Frozen data, geometry and execution semantics

Dataset `outputs/cm-interaction-oracle/per-finger-control-s227/interventions.pt`
SHA138b99b21e5567616dbdb6556179cc87dc8b42b578cad7e127dc6d5590e17149.
Oracle split `gt-consequence-s231-r2/diagnostic.pt` SHA16b6b0961c467e2836ca90cf0dc7285269570ac8fb1ed8129619e84091c63884.
Keep every window including early failures. E12/I14 atstep8, eight Y heads
at9..32. Full train-only H PCA32+physical72, common geometry PCA16; OOF H and
geometry PCA fitted only on source environments. Three folds seed245.

Reuse measured-root FK and same120 corresponding hand samples/seed241.
Use64 deterministic uniformly indexed points of saved1024object surface,
hash-checked against collection provenance. Current hand/object geometry,
normals and nominal baseline PD flow are decision-known. Actual future q,
PD targets or feedback sequences never enter model inputs. Nominal total
flow equals FK(target(base+delta))-FK(current_q). It interpolates a PD endpoint;
it is NOT a measured/predicted8-step execution trajectory. Duration8/30s is
a nominal feature convention, not a certified execution speed. Incremental
flow and native target perturbations retained for candidate contrasts.

Geometry is current object frame; E12/I14 compact labels contain world-axis
quantities. H includes decision-time object pose. No claim of equivariance.
State/Arm/Joint spatial inputs use nominal source action (arm0), not zero
total flow. All models also share nominal baseline geometry PCA state.
Flow/Shuffled spatial inputs use same recipient state with factual/permuted
intended arm. Shuffle within wave×motion×phase separately for source/hold/test,
seed246; never move donor geometry to recipient state. No PCA action whitening.

## Matched spatial training and OOF

Reuse ObjectInteractionCmv2V13Model fused_feature: local swept2cm KNN8,
competitive4tokens, cross-attention, physical20mm scale, hidden32.
This is a reduced spatial adaptation, not the original GRAB/MANO training
configuration or checkpoint. Compact persistence-residual head predicts26
standardized E/I values, MLP64/32tanh. E baseline0; I baseline current exact
physical readout. All target scales source-only. Same network widths/init
seed245 and minibatch schedule seed246 in State/Arm/Joint/Flow/Shuffled.
Fixed300 AdamW updates, batch64 with replacement, lr.001, wd.01, clip2.
No epochs appended after seeing results. Native Joint uses signed rad/.32,
Arm14onehot, common32action slots. Physical26slots zero for predictors.

Full outer train fivefits, three strict environment OOF folds fivefits each.
Source-fold state/PCA/target norms only. Downstream predictors train on OOF
consequences and test on full outer-train predictions. Eight separate same-
budget spatial task models H/Arm/Joint/Flow/GT/P_State/P_Flow/Flow_P_Flow;
common current persistence state provided to all, physical26slots zero or
GT/OOF values. Direct Flow task model uses actual spatial pathway and equal
training budget, avoiding credit for defeating prior unstable bilinear ridge.
GT is prognosis oracle, not an actionable decision-time input or mediation proof.

## Frozen decisions and evidence boundaries

A: Flow I gain over State≥5%, paired environment bootstrap lower95>0,
gain over trained Shuffled≥3%, frozen test-action shuffle penalty≥3%.
B: test adjusted-arm design rank14, same-H I vector corr≥.5, sign≥.65,
gain vs zero≥10%, arm-centered gain>0. Also report +/− pairs and all14 raw
26D candidate-minus-zero vectors; these aggregate GT estimates are not
per-state paired counterfactual truth. Geometry-specific: I gain≥3% over
Arm, Joint AND TrainMean. Report E/I and persistence/mean controls and OOF.
C: P_Flow or Flow_P_Flow primary gain≥5% over direct spatial Flow with
lower95>0 and physical-failure point gain≥−2%. Also require P_Flow vs
P_State primary gain≥5%, lower95>0 before calling unique action consequence
value. All five gates required for PROMISING; otherwise UNPROMISING with
rank14, UNCLEAR if contrast unsupported. An isolated C cannot rescue A/B.
Bootstrap246,2000environment resamples of fixed fitted models, no refits.
R oracle gain retention descriptive. No selector/PPO in this experiment.

## Artifacts, limitations / future evidence

Entry `tools/run/probe_spatial_consequence.py`; model `src/spatial_consequence.py`.
Unique run folder `outputs/cm-interaction-oracle/spatial-consequence-s245/`.
Save hashes, manifest, all28weights, normalization/folds/permutations, full/
OOF predictions,15same-state candidates, signed vectors, task readouts,
learning curves and existing per-finger PD/q/true-tip amplitude audit.
Inherited measured amplitudes are distinct from nominal surface endpoints.
Full-budget inference replay and independent abnormal-result review planned.
Future: independent holdout, actual execution prediction, denser object/hand
sampling, original full spatial capacity/pretraining, matched trained-policy
Cm-on/off Validation. These are deferred evidence, not immediate extra runs.
