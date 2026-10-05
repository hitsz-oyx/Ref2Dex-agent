---
schema: ref2dex.probe.v2
probe_id: P-20261005-spatial-consequence
experiment_id: P-20261005-spatial-consequence
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: 0f2480f
claim_id: C3
hypothesis_family: HF-spatial-consequence
probe_index_in_family: 1
seed_pool: probe
seeds: [245, 246]
decision_changed_if_positive: prioritize nominal spatial candidate mechanism with an independent frozen evaluation
decision_changed_if_negative: separate local spatial prediction from nominal execution limitations before collecting again
status: UNPROMISING
run_id: spatial-consequence-s245
---

# Does shared local spatial structure preserve intended action consequences?

Result: UNPROMISING: reduced spatial encoding remains stable but preserves too little intended action contrast; OOF consequences show no unique task gain.
Decision: Stop this fixed nominal spatial fit; retain GT prognosis and examine the nominal-to-realized execution contract before further geometry learning.

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

## Completed results and root interpretation

Execution commit0f2480f, GPU6, main162.05s (manifest162.62s), peak247.36MiB.
Saved-weight replay of all20consequence and8task models, all OOF
rows, frozen test shuffle and15Flow candidate arrays matches EXACTLY0;
regenerated current geometry also matches0. Audit is inference only3.44s.
All53Task tests pass. No simulation, selector, teacher or policy training.
Total new smoke+main artifacts about154MiB, within200MiB combined cap.

| Predictor | E12 MSE | I14 MSE | EI MSE |
| --- | --- | --- | --- |
| State | 0.50197 | 0.88370 | 0.70752 |
| Arm | 0.49635 | 0.87017 | 0.69763 |
| Joint | 0.49874 | 0.86704 | 0.69705 |
| Flow | 0.50390 | 0.88913 | 0.71133 |
| Flow_test_shuffled | 0.50688 | 0.89416 | 0.71541 |
| Shuffled | 0.50272 | 0.88715 | 0.70972 |
| TrainMean | 0.75352 | 0.89069 | 0.82738 |
| Persistence | 0.88281 | 1.04577 | 0.97055 |

A=false,B=false,geometry-specific=false,C=false,unique-over-state=false.
Flow I vs State gain−0.62%,95%−2.26..0.89%; vs Joint−2.55%,95%−4.87..−0.46%.
Test-action shuffle I error increases only0.565%,95%−0.840..1.872%.
Flow I is nearly train-mean, though E reconstruction beats mean33.13%.
That E gain is shared by State/Arm/Joint, not isolated action information.
Raw I vector corr.2392/sign53.14%/amplitude.0660/gain-vs-zero2.59%.
Centered gain4.20%, +/−pair gain3.23%, corr.1902/sign50.63%; neither rescues B.
All14signed E12/I14 vectors saved in per_arm_contrasts.json/Markdown/PNG.
GT arm design fullrank14; aggregate contrast uncertainty is still retained.

| Same-budget task model | Primary MSE | Physical failure MSE |
| --- | --- | --- |
| H | 0.54263 | 0.55399 |
| Arm | 0.50676 | 0.52225 |
| Joint | 0.51542 | 0.52635 |
| Flow | 0.54677 | 0.57788 |
| GT | 0.37866 | 0.49476 |
| P_State | 0.53292 | 0.55378 |
| P_Flow | 0.54419 | 0.57242 |
| Flow_P_Flow | 0.54710 | 0.57635 |

GT primary gain30.22%,95%11.98–43.15%; physical-failure gain10.69% but
95%−16.79..32.33%, so local GT prognosis remains useful without a new physical
safety claim. P_Flow vs directFlow gain0.47%,95%−8.31..7.79%; vs P_State
−2.11%,95%−4.75..1.00%; R−.0095,95%−.146.. .203, valid99.85%.
Adding predicted Flow to directFlow does not improve primary point error.
P_Flow has additional consequence pretraining/OOF cost, although downstream
budget is matched; this is not total-compute matched Cm-on/off policy evidence.

Train joint MSE State.6611/Arm.6432/Joint.6460/Flow.6531/Shuffled.6587;
Flow test.7113 and last50 minibatch loss.6942. This avoids the old nearly-zero
train/huge-test gap. Network remains capacity/update/regularization limited;
absence of strong contrast at300updates does not establish unlearnability.
The changed baseline and fitting contract, together with spatial encoding,
prevent attributing stable generalization exclusively to geometry.

## Independent abnormal-result review and root verification

At the user's request reviewer independently checked source-only statistics,
fresh PCA subspaces for full/threefold fits, OOF coverage and test replacement,
grouped permutations, world/object frame semantics, labels, all main metrics,
bootstrap and arm contrasts. Statistics differ≤1.91e−6 across CPU/GPU;
bootstrap numerical differences≤2e−7. Root replay regenerated FK and all28
models plus candidate predictions exactly; source/label hashes remained fixed.

99.06% factual windows have local edges; held test averages21.37/64 active
object queries. In nonzero treatment rows64.16%change spatial masks,
85.46%change KNN. Same saved Flow model test factual-versus-arm0 normalized
I RMS.03646; input-flow gradient norm.78907, edge parameter gradient.49375,
trained edge-weight displacement norm1.48109. The path receives actions and
learns; no missing-input or inert-network implementation explanation found.
OOF Flow I vs State improves only.47/.88/1.16% by fold, consistent with weak
conditional action information. Learning curves still fall: final50vsprior50
Flow loss−2.55%, so convergence/unlearnability is not established.

Root accepts review as implementation/limited-fit evidence, not a scientific
refutation. No added training/collection; independent audit2.35s, GPU portion
.74s. Reports copied without replacing originals to completed run as
independent_review.json/Markdown; audit_provenance.json hashes reviewer files,
root replay and audit code atcommitd059604. Source main manifest unchanged.

## Nominal versus realized execution boundary

Audit compares actual factual step8q/tips with decision-time nominal endpoint.
All true-tip vectors are in their respective measured hand-base frames;
source actor commands evolve during eight steps. It is not a matched zero
counterfactual, nor a causal decomposition of feedback compensation.
Zero-arm nominal vs actual mean tip travel(mm): index10.97 vs3.00,
middle9.53 vs3.38, pinky1.93 vs.96, ring4.81 vs2.32, thumb15.97 vs6.21.
Thumb-yaw-minus nominal thumb21.71 vs actual13.43, vector endpoint error9.57mm;
index-plus43.60 vs24.46, vector endpoint error19.64mm. Magnitudes and directions
can both differ; full15arm table and driver rad MAEs are in engineering_replay
and per_arm_contrasts Markdown. This makes the endpoint execution approximation
visible, but alone does not prove it caused the weak predictive action channel.

## Retained failures and engineering fixes

First smoke c88284c failed while hashing relative __file__, BEFORE GPU/model
computation; preserve spatial-consequence-smoke-s245/startup_failure.json and
its project tmp log. Fix3e08553 resolves path. Second smoke completed all28
one-update fits in6.17s, with full finite/FK/OOF checks; its original summary
contained an UNPROMISING diagnostic label. Independent review caught this
engineering-smoke semantic error; fix0f2480f forces smoke status UNCLEAR plus
engineering_only=true. The original smoke artifacts are retained as engineering
execution only and provide no scientific negative evidence. Main follows the
same predeclared300-update protocol and is unaffected.

## Root next Decision Note

Question: whether another nominal spatial fit or more collection can advance
MissionB after A/B/C failure. Evidence: stable errors, weak action contrast,
GT prognosis retained, nominal endpoint/live motion mismatch. Root stops this
specific fit without more epochs/seeds/radius/sampling scans. Next useful
Decision should distinguish predictable realized execution from contact-law
failure using existing nativeq/PD/tip records and a separately frozen contract;
future execution labels may be an oracle diagnostic but cannot enter a causal
selector. No policy work until action contrast and downstream value pass.
Expected next offline Probe remains bounded singleGPU; stop if it cannot
change acquisition-versus-modeling choice. No external authorization needed.
Cm hypothesis, Mission claim and matched trained-policy utility remain OPEN.

## Per-finger physical perturbation amplitude audit

The inherited measurement table is retained here for lookup as requested.
Nominal endpoint movement is separate from current-adjusted actual treatment
contrasts. All dose, follower, rad/deg and range metadata remain in JSON.

## Per-finger perturbation amplitude audit

Units: native action dimensionless; PD targets/actual q radians; tip displacement mm.
PD target changes are commanded, not actual executed joint angles. Driver table order: index, middle, pinky, ring, thumb-yaw, thumb-pitch. JSON PD/action arrays use all12 native joints6..17, including followers.

| Driver | Native index | Physical range (rad) | Native mimic followers |
| --- | --- | --- | --- |
| index | 6 | 1.6000 | 7: ×1.05 |
| middle | 8 | 1.6000 | 9: ×1.05 |
| pinky | 10 | 1.6000 | 11: ×1.05 |
| ring | 12 | 1.6000 | 13: ×1.05 |
| thumb_yaw | 14 | 1.1500 | none |
| thumb_pitch | 15 | 0.5500 | 16: ×0.6;17: ×0.8 |

### Delivered PD target magnitude per arm

Signed offsets per active step, not cumulative K-times angles. Finger targets are absolute native PD targets with a per-step baseline offset. Min/mean/max per joint, degrees and range fractions are preserved in `finger_amplitudes.json`. Composite arms exclude thumb-yaw. Clip % counts active commanded finger coordinates.

| alpha | Arm | n | index rad | middle rad | pinky rad | ring rad | yaw rad | pitch rad | clip % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | zero | 56 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | index_plus | 49 | +0.32000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | index_minus | 56 | -0.32000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | middle_plus | 51 | +0.00000 | +0.32000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | middle_minus | 63 | +0.00000 | -0.32000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | pinky_plus | 72 | +0.00000 | +0.00000 | +0.32000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | pinky_minus | 46 | +0.00000 | +0.00000 | -0.32000 | +0.00000 | +0.00000 | +0.00000 | 0.00 |
| 1 | ring_plus | 60 | +0.00000 | +0.00000 | +0.00000 | +0.32000 | +0.00000 | +0.00000 | 0.00 |
| 1 | ring_minus | 55 | +0.00000 | +0.00000 | +0.00000 | -0.32000 | +0.00000 | +0.00000 | 0.00 |
| 1 | thumb_yaw_plus | 49 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.23000 | +0.00000 | 0.00 |
| 1 | thumb_yaw_minus | 54 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | -0.23000 | +0.00000 | 0.00 |
| 1 | thumb_pitch_plus | 78 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.11000 | 0.00 |
| 1 | thumb_pitch_minus | 42 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | -0.11000 | 0.00 |
| 1 | synergy_plus | 62 | +0.32000 | +0.32000 | +0.32000 | +0.32000 | +0.00000 | +0.11000 | 0.00 |
| 1 | synergy_minus | 61 | -0.32000 | -0.32000 | -0.32000 | -0.32000 | +0.00000 | -0.11000 | 0.00 |

For composite ± at a20% driver range dose, intermediate followers6→7 etc receive ±0.336rad (10.70% of their3.14rad range); thumb-pitch followers receive ±0.066/0.088rad (2.10/2.80% of their ranges). At5% driver dose these are divided by4. Yaw has no target coupling.

### Measured native driver response at step8

Current-state-adjusted arm-minus-zero contrasts of actual q(step8)−q(before), radians. Not PD targets.

| Arm | index | middle | pinky | ring | yaw | pitch |
| --- | --- | --- | --- | --- | --- | --- |
| index_plus | +0.12875 | +0.03508 | +0.01204 | +0.00308 | -0.00562 | -0.00613 |
| index_minus | -0.15809 | -0.02648 | -0.00266 | +0.00306 | +0.00053 | -0.02202 |
| middle_plus | +0.05977 | +0.16934 | +0.02540 | +0.06805 | -0.00172 | -0.01241 |
| middle_minus | -0.03351 | -0.14832 | +0.00320 | -0.03896 | +0.00427 | -0.00623 |
| pinky_plus | +0.01215 | +0.02435 | +0.26734 | -0.00121 | +0.00015 | -0.00239 |
| pinky_minus | -0.00205 | +0.01421 | -0.26966 | -0.03733 | +0.00081 | +0.00415 |
| ring_plus | +0.00987 | +0.04957 | +0.01291 | +0.23303 | -0.00184 | -0.00111 |
| ring_minus | -0.00517 | -0.00878 | +0.00370 | -0.22309 | +0.00526 | -0.00427 |
| thumb_yaw_plus | -0.02100 | -0.01908 | -0.00647 | -0.01102 | +0.21367 | -0.00504 |
| thumb_yaw_minus | +0.00048 | +0.03201 | +0.00474 | +0.00392 | -0.20989 | -0.00512 |
| thumb_pitch_plus | -0.02185 | +0.00034 | +0.00140 | -0.00016 | +0.00352 | +0.08702 |
| thumb_pitch_minus | +0.01170 | +0.02278 | +0.00628 | -0.00391 | +0.00023 | -0.10148 |
| synergy_plus | +0.20111 | +0.25448 | +0.29047 | +0.25572 | -0.01000 | +0.08441 |
| synergy_minus | -0.21090 | -0.23692 | -0.30172 | -0.29002 | +0.00434 | -0.12145 |

### Measured five-finger tip response at step8

True tip positions in the measured hand-base frame. Values are norms of adjusted displacement vector contrasts (mm), not differences of average travel. Raw within-arm travel means are in JSON and include ordinary baseline evolution.

| Arm | index mm | middle mm | pinky mm | ring mm | thumb mm |
| --- | --- | --- | --- | --- | --- |
| index_plus | 21.69 | 2.43 | 0.28 | 0.12 | 1.05 |
| index_minus | 24.38 | 2.07 | 0.14 | 0.11 | 6.85 |
| middle_plus | 4.03 | 25.71 | 0.80 | 4.12 | 2.96 |
| middle_minus | 2.42 | 26.85 | 0.13 | 2.70 | 1.34 |
| pinky_plus | 0.80 | 1.61 | 15.19 | 0.05 | 0.48 |
| pinky_minus | 0.11 | 0.58 | 17.56 | 2.66 | 1.68 |
| ring_plus | 0.74 | 3.27 | 0.45 | 27.62 | 0.96 |
| ring_minus | 0.45 | 1.05 | 0.37 | 29.06 | 0.35 |
| thumb_yaw_plus | 1.55 | 1.74 | 0.21 | 0.92 | 9.33 |
| thumb_yaw_minus | 0.09 | 1.65 | 0.19 | 0.21 | 9.87 |
| thumb_pitch_plus | 1.48 | 0.26 | 0.27 | 0.42 | 10.15 |
| thumb_pitch_minus | 0.73 | 1.44 | 0.26 | 0.18 | 13.94 |
| synergy_plus | 25.86 | 30.97 | 15.86 | 29.79 | 8.58 |
| synergy_minus | 27.79 | 32.82 | 18.92 | 33.61 | 19.81 |
