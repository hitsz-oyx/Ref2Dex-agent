---
schema: ref2dex.probe.v2
probe_id: P-20261005-gt-consequence-sufficiency
experiment_id: P-20261005-gt-consequence-sufficiency
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: 08975e1
claim_id: C3
hypothesis_family: HF-gt-consequence-sufficiency
probe_index_in_family: 1
seed_pool: probe
seeds: [231, 232]
decision_changed_if_positive: prioritize action-conditioned consequence prediction on this early-hold representation
decision_changed_if_negative: distinguish GT prognosis from missing action-related information before further Cm fitting
status: UNCLEAR
run_id: gt-consequence-s231-r2
---

# Do GT physical consequences preserve task-relevant action information?

Result: GT prognosis PROMISING (46.25% primary error gain); held-arm EI bridge positive; full sufficiency UNCLEAR because remaining-action uncertainty exceeds 5%.
Decision: Preserve task-relevant E/I evidence and qualify the next conditional-predictability question; no claim of identified sufficiency, online selector or policy utility.

## Motivation and Decision Note

Ref8 asks to close action→(E,I)→Y on strong-action-response early-hold data.
This is a Decision for the Task's GT-usefulness-before-Cm subgoal, ultimately
serving trained-policy matched Cm-on/off utility. Reuse ref7; no new simulator
samples. Existing local ref7 joint gate remains UNPROMISING; this is a NEW
representation question authorized by ref8, not a retrospective gate rescue.
GT prognosis alone is insufficient: strong baseline failure state can predict
Y without capturing action-induced task variation. Small incremental action
benefit also does not identify causal mediation: randomized a does not randomize
post-treatment E/I; confounding, measurement error, weak action→Y, model error
and power remain. Neither model comparison nor arm regression proves sufficiency.

Root chooses six matched scorers H/Ha/HE/HI/HEI/HaEI plus three PREDECLARED
signed-force extensions HI_signed/HEI_signed/HaEI_signed. Main E12/I14 are
unchanged native readouts; signed extension adds asinh of 18 world net-force
components. No representation/epoch/seed search. Cheapest discriminator is
one fixed fit per scorer and CPU statistics on already saved observations.
One currently idle physical GPU6; timeout600s inclusive; outputs≤100MiB;
no simulation, Cm, selector, policy training or external modifications.
Stop on input/code drift, nonfinite values, environment overlap or resource
conflict; retain failed artifacts. Positive opens a separate consequence
predictability Probe; negative/unclear reports which link is unsupported,
without abandoning the core claim. Mission/Campaign boundaries unchanged.

## Frozen protocol before outcomes

Input `per-finger-control-s227/interventions.pt`, SHA256
`138b99b21e5567616dbdb6556179cc87dc8b42b578cad7e127dc6d5590e17149`.
Original collection commit a81b36d, 15arms/K8/20% driver range, all854 complete
windows including193 early failures. Recheck dose/geometry and attach the full
finger PD/q/tip amplitude table to this card. Readout E12/I14 and signed18
only at step8; Y ONLY step9..16/32. Six original continuation heads, plus
late physical height-failure (<2cm once) and height-held fraction (≥3cm).
All risk trials retained; no future survival filter or Y in H.

Environment cluster=episode_id modulo168; motion-stratified80/20 split seed231;
all waves for an environment stay on one side. H=train-only standardization
and PCA32 of historical1390/actor1442/context435/base18, CONCATENATED with
train-standardized current physical72 to protect explicit present-state
information from PCA compression. Action=14 nonzero-arm one-hot indicators;
no post-treatment realized action/PD/baseline-feedback sequence as an input.
E/I/signed and all8 Y heads standardized using train only (scale floor.001).
Zero padding gives all9 models identical input width, parameters and initial
weights, seed231, MLP64/32 tanh, AdamW lr.002/decay.001, clip2,
200 full-batch epochs, no validation selection. Save train loss traces,
normalizers/PCA/split/weights/predictions. Last50 loss change reports fitting
limits, not convergence proof.

Primary normalized error averages contact32, late physical height failure,
height-held fraction. Report all8 heads, physical failure AUC/Brier and
factual cross-arm prognosis ranking within motion×phase quarter. Ranking
includes ONLY strata with≥10 eligible pairs; not same-state action ranking.
Paired bootstrap2000 on held-out environment clusters seed232 gives fixed-fit
95% intervals; it does not include training-seed variation.

Support: ≥120 test windows, ≥25 test environments, ≥5 test trials per motion,
≥15 height failure AND nonfailure test cases, ≥3 held-out trials for EACH arm.
GT prognosis PROMISING if HEI vs H primary gain≥10%, physical-failure gain≥5%
and both bootstrap lower95>0; UNPROMISING if support is adequate but these
fail, otherwise UNCLEAR. Extensions do not rescue the main registered label.
Conditional action increment is small only when HaEI vs HEI primary AND
physical point gains≤2% and upper one-sided95≤5%. Report HI/HE, Ha effects
and the signed extensions with identical comparisons, without significance
claims from picking the best scorer.

Action-related check: independently residualize joint [E12,I14,signed18,Y8]
on the original CURRENT-ONLY nuisance controls and arm indicators, full cohort
and two five-wave halves. Preserve signed/vector contrasts and all14 arms.
Cross-half, leave-one-arm-out transfer fits only13 source-half arm contrasts,
uses feature scaling and uncentered PCA3 learned on those13, ridge1 without
intercept, predicts held arm in the OTHER half. Report E/I/EI/EI_signed and
zero/source-mean baselines. No in-sample 14-arm many-feature fit. Descriptive
bridge positive requires EI transfer contact32 AND physical-failure gains
vs BOTH baselines≥10% with both correlations≥.30. This coarse bridge is
not a mediation estimator; shared-zero/noisy contrast uncertainty remains.

Overall chain PROMISING only if adequate support, main GT prognosis positive,
small remaining-action increment AND the EI arm bridge is positive. Otherwise
keep per-link labels: absent GT gain is UNPROMISING for this representation;
positive prognosis without action bridge is UNCLEAR for actionable sufficiency.
This cannot form a formal scientific conclusion or complete policy utility.

## Limitations and future evidence

Single collection and fit seed, official frozen source actor used only as
research substrate. Current I aggregate net-force/proximity≠paired contact,
slip or friction margin. E8 may reflect failure already beginning; retain all
trials, disclose early-failure sensitivity descriptively only. Negative arm
bridge may reflect noisy14-arm estimates or nonlinear/state-specific effects.
A richer H, model-capacity/fit adequacy, independent validation, matched
same-state candidate comparisons and trained-policy Cm causal utility remain
future evidence if a concrete decision requires them; no sweep by default.

## Engineering execution record

Initial source d8681aa run `gt-consequence-s231` failed before ANY neural fit:
CUDA memory-stat reset occurred before explicit device initialization (0.76s).
CPU label/contrast/support outputs are preserved, no result/model generated.
Explicit `torch.cuda.set_device(cuda:0)` smoke passes on GPU6; repair does not
change scientific variables, labels, split, epoch budget or gate. Continue in
new directory `gt-consequence-s231-r2`, never overwrite the failed attempt.

## Completed results and root interpretation

Actual source08975e1, run `gt-consequence-s231-r2`, fixed fit seed231:
688train/166test windows,125/31 disjoint environment clusters. Test motions
100/45/21, all15 test arms have3–16trials;90 physical height failures/76
nonfailures.33 test early failures retained (193 total). Support PASS.
All9 identical162-input/12,776-parameter scorers share initial weight hash,
200epochs, no checkpoint/test selection. Original dataset hash unchanged.

| Model | Primary normalized test MSE | Late physical-failure normalized MSE | Physical-failure AUC | Primary gain vs H |
| --- | --- | --- | --- | --- |
| H | 0.731350 | 0.728881 | 0.8497 | 0.00% |
| Ha | 0.592461 | 0.583994 | 0.8779 | 18.99% |
| HE | 0.478871 | 0.586646 | 0.8787 | 34.52% |
| HI | 0.411747 | 0.515434 | 0.9088 | 43.70% |
| HEI | 0.393108 | 0.491827 | 0.9028 | 46.25% |
| HaEI | 0.394387 | 0.496376 | 0.9020 | 46.07% |
| HI_signed | 0.430010 | 0.537879 | 0.8990 | 41.20% |
| HEI_signed | 0.425418 | 0.554856 | 0.8816 | 41.83% |
| HaEI_signed | 0.388489 | 0.483231 | 0.9020 | 46.88% |

GT E/I prognosis **PROMISING**:primary error reduction46.25%, environment
bootstrap95% interval34.02–57.03%; physical-height-failure reduction32.52%,
interval11.24–49.92%. GT I alone gives43.70% primary reduction; E alone34.52%.
This cohort therefore preserves the earlier useful-I prognosis under a much
stronger intervention contract. H+a itself improves primary error18.99%
(interval5.03–32.27%) and physical error19.88%(1.64–36.71%). Global marginal
height-response family in ref7 was weak; this state-conditioned prediction
comparison is a different estimand and is not a causal effect proof.

HaEI vs HEI point gains:primary−0.325%, physical−0.925% (slightly worse),
but primary95% interval−10.96–9.51%, physical−14.97–11.79%; one-sided95%
upper bounds7.96/9.64% exceed the registered5%. Thus **small remaining-action
criterion FAILS on uncertainty**, not because a substantial missing action
channel was established. No retrospective relaxation or equivalence claim.

Registered cross-half held-arm EI bridge passes contact and physical failure:

| Consequence features | Contact gain vs zero / source mean | Height-failure gain vs zero / source mean | Contact / height correlation |
| --- | --- | --- | --- |
| E | 8.25% / 17.61% | -8.41% / 43.63% | 0.2883 / 0.1821 |
| I | 10.43% / 19.58% | 4.19% / 50.18% | 0.6626 / 0.7577 |
| EI | 21.65% / 29.65% | 16.14% / 56.40% | 0.5503 / 0.7233 |
| EI_signed | 11.35% / 20.40% | 33.49% / 65.42% | 0.3859 / 0.7644 |

These28 folds predict estimated contrasts, not28 independent experiments.
Each half's nuisance fit includes all its arms; bridge PCA/ridge excludes the
held source arm, but raw-state nuisance estimation is not completely held-arm
cross-fitted. Shared-zero/noisy contrast dependence remains. EI height-held
FRACTION gain vs zero is−18.11%, although height-failure gain ispositive:
task heads are not interchangeable, and the bridge does not capture every
physical outcome. Signed extension improves the coarse bridge height-failure
gain (33.49%) but does not uniformly improve individual prognostic errors.

Signed extension:HEI_signed primarygain41.83% (less than46.25% main); adding
a now improves primary8.68%(95%3.45–14.56%) and physical12.91%(3.75–23.41%).
This sensitivity prevents claiming a robust representation-equivalence result,
and gives no basis to say force norm compression is the main missing channel.
A richer feature set can fit differently at fixed data/budget.

Supported-stratum factual prognosis ranking H→HEI:contact69.55→78.54%,
physical-failure76.94→87.87%, height-held fraction64.35→84.70%; tiny strata
excluded as registered, denominators4/3/4 strata. HaEI contact ranking84.36%
versusHEI78.54% despite similar primary MSE: absent aggregate MSE benefit
does NOT establish identical ordering. These cross-state pairs are not
same-state candidate action ranking; no deployable selector was evaluated.

Descriptive no-early-failure sensitivity:133test windows,57 subsequent height
failures; H primaryMSE0.78789→HEI0.45831 (41.83% gain), so GT benefit is not
confined to already-failed trials. This post-treatment subset does not replace
the randomized cohort or identify causal mediation.

All models last50 train losses still fall38–46%; train ALL8-head MSE ranges
0.0206–0.0719, versus matched test ALL8-head0.3494–0.7332. No convergence
claim. Physical72 is explicit in H, while rawq/dq/true-tip/base information
is chiefly in history/actor PCA; compressed initial-state information and
fixed-fit capacity/optimization remain possible confounders of comparisons.
Same-primary train/test: H0.06620/0.73135, HEI0.03591/0.39311,
HaEI0.02424/0.39439. Do not call absence of an incremental fitted a effect
proof of sufficiency.

Root judgment: there is now useful evidence linking strong action→I, useful
GT E/I prognosis, and out-of-half signed/vector consequence-to-task contrasts.
The registered full chain remains **UNCLEAR**, with GT prognosis **PROMISING**;
this is progress beyond prognosis alone, not a validated mediator claim.
The next useful Decision is CONDITIONAL PREDICTABILITY:can H+a predict these
useful E/I, and can predicted consequences retain task gain against the
matched direct H+a scorer? That question can be worthwhile before causal
sufficiency is established, but must be a separately frozen bounded Probe,
include direct Ha, and address representation sensitivity; it was NOT run
here. Do not collect more samples solely to narrow the5% interval or launch
a selector/PPO from oracle inputs. North-star trained-policy Cm utility OPEN.

## Signed/vector arm contrasts retained

All14 full signedE12/I14/signed18/Y8 vectors and both half estimates are in
`arm_contrasts.json`; the table below is a readable subset, not norm-only
interaction analysis. Ownbody=isolated finger, thumb for yaw/pitch; synergy
shows thumb only here, with all5body vectors retained in JSON. Contrasts are
current-state-adjusted marginal arm-minus-zero estimates, not certified force,
load attribution or same-state mediation.

| Arm | E dz mm | E delta vz m/s | Own log1p force norm | Own surface distance mm | Late contact pp | Late height-failure pp |
| --- | --- | --- | --- | --- | --- | --- |
| index_plus | -6.880 | -0.1539 | +0.1187 | +0.130 | -1.44 | +3.74 |
| index_minus | +1.784 | +0.0044 | +0.0437 | -1.398 | +0.45 | -4.40 |
| middle_plus | +2.343 | -0.0407 | +0.0834 | -0.540 | +4.50 | -5.74 |
| middle_minus | -18.433 | +0.0429 | -0.2806 | +11.248 | -10.94 | +6.23 |
| pinky_plus | -9.446 | -0.0253 | -0.0761 | +7.449 | +4.10 | -4.56 |
| pinky_minus | -3.113 | +0.0444 | -0.1345 | +7.389 | -0.57 | +1.43 |
| ring_plus | -9.737 | -0.0014 | -0.0667 | +1.243 | +2.30 | -4.13 |
| ring_minus | +4.940 | +0.0518 | -0.0930 | +2.302 | -0.95 | -9.53 |
| thumb_yaw_plus | +3.383 | +0.1503 | +0.1064 | -1.070 | +5.61 | -3.70 |
| thumb_yaw_minus | -0.785 | +0.0869 | -0.0791 | -1.154 | +10.76 | -11.34 |
| thumb_pitch_plus | -1.588 | +0.0351 | +0.1063 | -0.921 | +5.75 | -6.31 |
| thumb_pitch_minus | +6.840 | +0.0262 | -0.0402 | -2.271 | +7.89 | -5.07 |
| synergy_plus | -6.330 | -0.0080 | +0.1034 | -3.908 | -3.38 | +4.29 |
| synergy_minus | -1.953 | +0.0573 | -0.5784 | +26.765 | -12.60 | +6.99 |

## Audit, artifacts and resource completion

Root independently loops contact runs acrossstep8 and all continuation/height
labels (maxerror1.99e−8), quaternionE3.05e−7, I/signed2.39e−7. Saved-prediction
metrics match≤5.96e−8, separate environment-bootstrap≤2.28e−7, explicit
joint-design OLS≤2.76e−9, augmented-least-squares bridge≤7.64e−16. Normalizers
use train only; env overlap0 and9 initial hashes identical. Float32 long GPU
SVD projection orthogonality error4.70e−4 is disclosed, within dimension-aware
1.57e−3 audit envelope; projected channels are subsequently train-standardized
and shared by all models. No result/data/code mutation or model rerun follows
these checks. Root audit script's initial syntax/tolerance checks were corrected
before a PASS artifact was emitted; no scientific fit affected.

Independent read-only reviewer separately reconstructs labels, train-only
normalizers, held-out predictions, paired bootstrap and bridge via Gram-eigen
PCA rather than production SVD; numerical differences≤7.7e−16 for bridge,
≤2.3e−7 for bootstrap. Reviewer confirms no time leakage, and flags shared-zero
nuisance, incomplete fit and compressedH limits. Root accepts those constraints;
review/testing cannot establish causal sufficiency.

Actual GPU6 run5.56s inclusive (first failedpre-fit0.76s); peak neural
allocated61.44MiB; all GPU work ended and GPU6 idle. New artifacts<4MiB.
No simulation/checkpoint overwrite, no changes to prior negative gates.
41 Task tests and scoped repository verification pass.

Outputs `outputs/cm-interaction-oracle/gt-consequence-s231-r2/`: immutable
manifest/result/diagnostic, arm_contrasts/arm_bridge, loss_traces, source/dose
engineering_audit, statistical_replay, independent_engineering_review,
finger_amplitudes JSON/Markdown and `gt_consequence_sufficiency.png`.
Failedpre-fit `gt-consequence-s231/` remains unchanged. Reproduce fit using
`CUDA_VISIBLE_DEVICES=6 TMPDIR=$PWD/tmp OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`
with Task `tools/run/probe_gt_consequence_sufficiency.py --dataset`
`outputs/cm-interaction-oracle/per-finger-control-s227/interventions.pt`
`--run-dir <new-unique-dir>`; resource cap must still apply. File/statistics
replay uses `tools/audit/audit_gt_consequence_sufficiency.py` with the same
arguments, no neural inference. Do not overwrite existing replay artifact.

The following amplitude audit reuses EXACT ref7 dataset and current-only
adjustment; no new perturbations or physical trajectories were collected.
Raw min/mean/max delivered PD, actual12-joint q and true5tip travel are in
JSON; tables retain driver/mimic and adjustedvector distinctions for lookup.

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
