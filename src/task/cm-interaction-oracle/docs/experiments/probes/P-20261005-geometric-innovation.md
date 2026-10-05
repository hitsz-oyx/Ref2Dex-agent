---
schema: ref2dex.probe.v2
probe_id: P-20261005-geometric-innovation
experiment_id: P-20261005-geometric-innovation
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: f64b681
claim_id: C3
hypothesis_family: HF-geometric-innovation
probe_index_in_family: 1
seed_pool: probe
seeds: [241, 242]
decision_changed_if_positive: prioritize intended geometric consequence modeling before candidate mechanism and matched trained-policy utility
decision_changed_if_negative: distinguish baseline or continuous-action gains from geometry gains without closing the spatial Cm hypothesis
status: UNPROMISING
run_id: geometric-innovation-s241-r2
---

# Ref10: does intended surface motion generalize beyond action categories?

Result: UNPROMISING: fixed PCA/bilinear ridge nominal-flow screen extrapolates badly; independent review plus frozen replay find no fatal wiring bug, and a separate support factorial isolates amplification without showing geometry superiority.
Decision: Stop the unconstrained PCA/bilinear endpoint route, retain physical action contracts and local spatial/execution hypotheses; do not enter selector or call a gain against collapsed direct Flow unique Cm utility.

## Motivation and root Decision Note

This Decision serves Mission C3, GT→predictable consequence→trained-policy
utility. Ref9 memorized training data yet lost to train-mean/persistence on
held environments; its categorical PCA/MLP failure does not test the original
OI-CmV2 geometry architecture. Ref10 explicitly authorizes intended FK point
flow and physics innovation. Question: does geometric action inductive bias
improve held-environment I/action contrasts over both categorical and native
continuous controls, independently of a stronger baseline/fitting contract?

Cheapest discriminator uses the same854 ref7 windows, ref8 train688/test166,
125/31 environments, E12/I14 step8 and eight Y9..32 heads. All failures stay.
Root chooses a fixed exact ridge on physical surface motion before a larger
spatial network: this isolates representation cheaply and makes optimization
error irrelevant. This is a geometric-input Probe, NOT the original spatial
OI-CmV2 network, NOT ref5 point-flow output E, NOT policy utility. A negative
here cannot close the core Cm/point-network hypothesis. A positive warrants
a separately frozen spatial/candidate mechanism experiment; not immediate PPO.

Resource: idle GPU6, timeout600s, outputs≤100MiB; tiny synthetic FK engineering
tests on CPU because GPU startup dominates. Statistical OLS/bootstrap on CPU;
all real FK/PCA/predictor/scorer fits and inference on GPU. Stop on source/data
drift, nonfinite, FK mismatch>0.1mm, environment/OOF leakage or resource conflict.
User requires independent review for anomalous results: reviewer checks native
PD/FK contract now and actual artifacts after the run. No external authorization
needed; Mission, claims and resource boundaries unchanged.

## Decision-time physical action contract

Native q=history[-1,:18] includes actual measured followers; base_action is
the decision-time source actor action. Exact DExplore clamping/scales/mimic
coupling generates all15 PD target candidates; all15 have preassignment
headroom. FK transforms include measured actor hand_root, preserving the wrist
chain. Compare FK current handbase/true tips to before measurements. Sampling
is area-based then deterministic24points per finger,120 total, covering the
proximal/intermediate/distal regions represented in I. Seed241; saved local
points/link IDs guarantee correspondence across candidates.

F_total(a)=P(FK(target(base+delta_a)))-P(FK(current_q)). Baseline F_total(0)
is generally NONZERO and shared as state by every arm. Geometric action is
delta_F(a)=F_total(a)-F_total(0). Express point/vector in decision object frame;
include signed delta_F dot current surface normal. No actual future q, tips,
actions, PD targets or feedback sequence enters inputs. Nominal endpoints are
NOT an executed eight-step sweep and are NOT multiplied by8. Decision-time
base action is fixed in representation; actual feedback base_t varies over8
steps and remains unmodeled. World-axis before state remains available for E.

Baseline E: constant world linear/angular velocity for8control_dt, short
rotation representative, zero velocity innovation. Baseline I: exact current
force-log/proximity/global-proxy features. Raw target minus this baseline is
a statistical innovation, not an identified causal treatment effect.

## Frozen matched fitting and OOF contract

Shared H104 retains ref8 exact full-fit PCA basis + current physical72;
OOF fits ref9 helper's PCA only on fold training environments. Shared geometry
state=(current120surface positions/normals + baseline nominal point flow),
train-only PCA16. Action32slots: zeros State; standardized14onehot Arm;
standardized18PD_target(candidate)-PD_target(base) Joint, padding zeros;
Flow PCA32 of120point incremental xyz +120normal components. Fit geometry
PCA on raw metre coordinates; no pre-PCA per-point whitening. All projected
scores normalized on fit rows only. State-action products=H's first16 ×32
action slots; same664 design columns for every predictor. Fixed α32 exact
centered float64 ridge, unpenalized intercept. Target residual scaled by
fit-only raw E/I std (floor.001). RawState/RawArm same x, ridge, target scale
predict raw E/I, isolating baseline subtraction from representation changes.

FlowShuffled assigns a permuted arm within wave×motion×phase-quarter,
separately fit/hold/test, seed242. It REGENERATES flow at the recipient state's
q/base/root, rather than moving a donor's state-dependent geometric feature.
Also test frozen Flow with permuted test arms. PCA fitted on factual fit rows
and reused for candidate/shuffle; held data never changes it. No tuning.

Three environment OOF folds seed241 inside outer train; fold H/geometry/action
PCA/norms/target scales fitted only other-fold environments. Save source/hold,
permutation, weights, normalizers, predictions. Downstream train always OOF,
test full-fit. Same α32 scorer: common H plus decision-known physics baseline
for ALL arms, matched action/product26consequence slots. H/Arm/Joint/Flow,
GT, P_State, P_Flow, Flow_P_Flow. Main comparison P_Flow or Flow_P_Flow against
direct Flow, not merely weak H or previous overfit Ha. Report all task heads.

## Predeclared decisions and evidence boundaries

A: Flow I gain over State≥5% and environment-bootstrap lower95>0;
also Flow I gain over trained FlowShuffled≥3% and frozen Flow test-action
shuffle error penalty≥3%. These latter two are point-estimate Probe screens;
their confidence intervals must also be reported, not silently upgraded.
B: same-state15candidate contrasts against ref8 current-adjusted factual GT
fulltest rank14, raw I corr≥.5/sign≥.65/gain-vs-zero≥10%, AND arm-centered
I gain-vs-zero>0. Also report seven within-pair +/− contrasts to cancel shared
zero offset. Unlike ref9 this shared-reference sensitivity is PREDECLARED.
Geometry-specific: Flow I gain≥3% over BOTH Arm and Joint. Report paired CI,
do not interpret marginal significant comparisons as definitive model ranking.
Representation PROMISING only A+B+geometry-specific; otherwise UNPROMISING
with rank support (or UNCLEAR without it). This is a fixed linear-feature
screen, not a closure of nonlinear spatial modeling. C unique transfer:
P_Flow or Flow_P_Flow ≥5% primary gain over direct Flow with lower95>0,
and physical-failure gain≥−2% (point-estimate safety screen, not certified safety).
No selector unless separate mechanism evidence, C and further validation.
Report E12/I14/joint MSE vs TrainMean, Persistence, SimpleDynamics, RawArm;
OOF quality, train/test gap, bootstrap CIs and same-budget GT retained R.
Bootstrap242,2000 environment resamples, fixed fitted models, not refitting CI.

## Results and review

Main run sourcef64b681, successful GPU6 fit2.62s, peak247.36MiB. Original
5352fc1 run failed before learning because resolved visual meshes lie behind
authorized third_party symlinks; preserve failed folder/log and use absolute
mesh SHA identities. Same protocol/seed, no test result existed before restart.
Four new contracts and all50Task tests pass. Independent reviewer caught the
short-rotation issue before the run:88/854 naive ω8dt magnitudes exceedπ.
Actual short-vector baseline and root×FK implemented before code checkpoint.
Current true-tip mismatch max0.02989mm; explicitroot and full18q preserved.

| Predictor | E12 MSE | I14 MSE | EI MSE |
| --- | --- | --- | --- |
| TrainMean | 0.75352 | 0.89069 | 0.82738 |
| Persistence | 0.88281 | 1.04577 | 0.97055 |
| SimpleDynamics | 1.89212 | 1.04577 | 1.43639 |
| RawState | 0.80538 | 1.04749 | 0.93575 |
| RawArm | 1.08081 | 1.34941 | 1.22544 |
| State residual | 1.60434 | 1.07388 | 1.31871 |
| Arm residual | 2.45557 | 1.62547 | 2.00859 |
| Joint residual | 1.86900 | 1.28513 | 1.55461 |
| Flow residual | 13.90584 | 12.44019 | 13.11664 |
| Trained FlowShuffled | 6.62758 | 6.87675 | 6.76175 |
| Frozen Flow/test-shuffle | 5.74731 | 5.82651 | 5.78996 |

A=false,B=false,geometry-specific=false. Raw Flow I contrasts corr.06579,
sign45.71%,amplitude.32789,gain-vs-zero−8.26%; arm-centered gain−32.91%,
plus-minus gain−57.84%. These are aggregate contrast checks, not per-state
identified counterfactual ground truth. All14×26signed raw/normalized vectors
are exported, including GT and all predictors. No main-result rescue.

| Matched scorer | Primary MSE | Physical-failure MSE |
| --- | --- | --- |
| H with known baseline | 0.65400 | 0.61365 |
| Direct Arm | 0.97000 | 1.23025 |
| Direct Joint | 0.90584 | 0.91806 |
| Direct Flow | 2.31975 | 1.39982 |
| GT E/I | 0.44800 | 0.48093 |
| P_State | 0.61321 | 0.58366 |
| P_Flow | 0.71489 | 0.76793 |
| Flow+P_Flow | 1.90034 | 1.40872 |

Literal C=true: P_Flow beats unstable directFlow by69.18%,95%48.71–77.48.
Root interpretation explicitly rejects treating this as unique Cm gain:
P_Flow is worse than H by9.31%(CI−33.13..10.96% gain) and worse than P_State
by16.58%(CI−31.29..−3.24% gain). R=−.29555,CI−2.932.. .231,valid99.9%.
GT gain31.50%(CI8.32–51.03) still supports predictive physical supervision
as a local oracle, not action selection or policy utility. The isolated C
screen is a necessary comparison against direct geometry, not sufficient.

## Independent anomaly review and root attribution

At user request reviewagent independently reconstructed current PD/FK,
preprocessing/folds/ridge equation, CV baseline, prediction/candidate/scorer
arrays on CPU without new fitting. No fatal wiring/leakage/units/solver bug.
Float32 cross-device replay differences≤1.84e−4 rawunits; normal equation
relative residual≤6.76e−7. Root GPU replay ALL28predictors,15Flow candidates,
eight scorers is exactly0. This validates saved execution, not all hypotheses.

Flow I top1/top5 rows contribute57.84%/85.13% of total error; row186 env28
synergy-minus alone has normalized I1194.39. KEEP all outliers. Fulltrain
Flow PCA scales68.11..1.665mm; no component hits1mm floor, so no divide-zero
normalization bug. PC30 teststd/trainstd3.051,max26.08σ; action norm maximum
train17.13/test37.59. Product-block rownorm maximum100.42/627.13;
total design coordinate max30.43/251.74. Reviewer separately decomposes
interaction prediction contribution RMS train.794/test3.191. These are
supported-input extrapolation and multiplicative amplification signals.

Root introduced the unconstrained bilinear feature family in this cheap screen;
its failure belongs to that learning contract, not the user's physical
point-flow idea. Separate frozen [support factorial](P-20261005-geometric-support.md)
removes products and physical rescaling; I12.44→1.25→1.14, but latter still
loses to additive Joint1.04. This diagnoses an expensive-to-interpret failure
without deleting data or silently refitting main gates. Fixed CV baseline
also worsens E versus raw controls; constant velocity is a weak contact-state
prior here, not a reason to infer missing E information.

## Root next decision

Stop this fixed PCA/bilinear endpoint screen. Preserve explicit current-state
FK, intended nominal incremental surface input, train-only preprocessing and
OOF scaffolding. A further geometry experiment must use local contact-relative
surface structure/shared spatial inductive bias and a defensible execution
contract; reuse of per-PC whitening or unconstrained H×F products is unwarranted.
Nominal target motion and measured realized step8 motion remain distinct.
Do not spend more on this ridge model, add epochs, tune on the exposed test,
or advance selector/PPO. Original OI-CmV2 spatial network and trained-policy
matched Cm-on/off objective remain OPEN; this is useful route exclusion only.

## Artifacts and limitations / future evidence

Completed outputs: `outputs/cm-interaction-oracle/geometric-innovation-s241-r2/`;
failed startup: `outputs/cm-interaction-oracle/geometric-innovation-s241/`.
Completed manifest/result/fit_records/diagnostic, engineering_replay,
independent_engineering_review JSON/Markdown, per_arm_contrasts JSON/Markdown,
finger_amplitudes JSON/Markdown, head_metrics.json (all26raw axis MAE, eight
task-head MSE and OOF quality), run.log and geometric_consequence.png retained.
Independent report includes descriptive excluded-error diagnostics solely for
attribution; none are primary scores or used to remove factual windows.
Labels/persistence/GT definition unchanged. Single split and fixed α; ridge
effective capacity differs across representations despite identical slots.
Per-finger equal sampling and PCA can discard local contact information.
Different real feedback trajectories/unknown material and load not modeled.
Spatial token network, actual execution prediction, held-sign/finger transfer,
matched trained-policy Cm-on/off and multi-seed Validation deferred until
they can change the next research decision. Original OI-CmV2 remains open.

## Per-finger physical perturbation magnitudes (retained in card)

The measured ref7 audit below is reproduced for lookup. Nominal surface
mean displacement (24points/finger) is a different quantity from measured
step8 true-tip displacement, which uses current-adjusted treatment contrasts.

### Decision-time nominal surface endpoint displacement

| Arm | Index mm | Middle mm | Pinky mm | Ring mm | Thumb mm |
| --- | --- | --- | --- | --- | --- |
| index_plus | 10.151 | 0.000 | 0.000 | 0.000 | 0.000 |
| index_minus | 10.227 | 0.000 | 0.000 | 0.000 | 0.000 |
| middle_plus | 0.000 | 11.478 | 0.000 | 0.000 | 0.000 |
| middle_minus | 0.000 | 11.683 | 0.000 | 0.000 | 0.000 |
| pinky_plus | 0.000 | 0.000 | 10.186 | 0.000 | 0.000 |
| pinky_minus | 0.000 | 0.000 | 10.469 | 0.000 | 0.000 |
| ring_plus | 0.000 | 0.000 | 0.000 | 13.298 | 0.000 |
| ring_minus | 0.000 | 0.000 | 0.000 | 13.805 | 0.000 |
| thumb_yaw_plus | 0.000 | 0.000 | 0.000 | 0.000 | 7.452 |
| thumb_yaw_minus | 0.000 | 0.000 | 0.000 | 0.000 | 7.452 |
| thumb_pitch_plus | 0.000 | 0.000 | 0.000 | 0.000 | 5.083 |
| thumb_pitch_minus | 0.000 | 0.000 | 0.000 | 0.000 | 5.100 |
| synergy_plus | 10.151 | 11.478 | 10.186 | 13.298 | 5.083 |
| synergy_minus | 10.227 | 11.683 | 10.469 | 13.805 | 5.100 |

Baseline nominal surface shift mean50.08mm; zero intervention has nonzero
total source-action flow, but incremental candidate-zero is exactly0.
Commanded physical PD offsets: ±.320rad for four drivers, ±.230yaw,
±.110pitch. Measured q/tip contrasts are not model inputs.

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


Root independently verified review report: the separate six-fit factorial was
precommitted with pending results (5a50f1e), and NumPy weight replay≤3.06e−5
rawunits, ridge equation relative residual≤2.36e−7, all2000-bootstrap gains
match exactly. It is post-result diagnosis, not independent Validation.
