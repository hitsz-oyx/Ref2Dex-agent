---
schema: ref2dex.probe.v2
probe_id: P-20261005-relative-finger-innovation
experiment_id: P-20261005-relative-finger-innovation
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: 8809fee8715784ed5362886a21334ec3bbcabc60
claim_id: C3
hypothesis_family: HF-relative-finger-innovation
probe_index_in_family: 1
seed_pool: probe
seeds: [245, 246, 251]
decision_changed_if_positive: qualify global anatomical finger flow and action-centered learning for strict consequence OOF
decision_changed_if_negative: distinguish the fixed geometry representation from state-residual learning and avoid repeated fitting
status: UNPROMISING
run_id: relative-finger-innovation-s251
---

# Can full finger flow learn action innovations over an OOF state forecast?

Result: UNPROMISING for the fixed anatomical summary / OOF-state innovation contract.
Decision: Stop this fixed seven-head contract; retain the physical action and Cm hypotheses.

## Root Decision Note and motivation

Ref10/MissionB needs continuous physical action→E/I→policy value. Latest
fidelity Probe verifies only fixed-arm linear recoverability, not physical
generalization; its LocalFlow proxy cannot isolate radius loss. V13 was
designed for global object rigid effects, while I14 has ordered finger fields.
Question: does full per-finger geometry preserve useful action contrast when
the predictor no longer has to relearn common state evolution? Cheapest
intervention uses existing854windows, source/OOF State nuisance predictions,
and seven same-budget small heads. No new simulation, spatial width/epochs or
radius scan. Finger-global geometry and centered objective each get matched
controls; improvement cannot be attributed to two changed variables blindly.

Root chooses source-state residual plus candidate-centering and anatomical
flow because these directly address current missing-finger-interface/common
state competition hypotheses. GPU6 main≤600s, smoke≤120s, combined new
outputs≤40MiB. Models/FK/inference GPU; CPU bootstrap/reporting and tiny
contract tests (GPU startup dominates). Stop on hash/input drift, leakage,
nonfinite, nuisance replay>1e-5, resource conflict or artifact bound. No
external authorization, core claim or long-term identity change.

## Immutable state nuisance and causal input

Same688train/166test environment split, same E12/I14step8 and source scales.
Use prior spatial State's three environment-OOF predictions as train mu(H);
outer-test uses its full-source prediction, already verified identical in the
saved OOF array. Train residual target (z−mu_OOF)/scale. Existing State full
and source fold weights/normalizers bind the provenance. Do not fit another
state model or read actual future geometry as action input.

Common H is source's120dim history/current physical/nominal geometry state,
plus currentfingerq12, intended zero-arm fingerq12 and predicted zero-arm
fingerq12, all divided by.32rad and source-only standardized. All seven heads
get identical common input so flow has no unshared baseline-angle side channel.
Forecast train uses previously corrected execution source OOF; test full fit.

Nominal/forecast candidate q produce finger-only flow in the recipient
zero-arm hand-base frame: hold wrist6 identical to that recipient arm0 wrist,
FK all15candidates, subtract arm0 endpoints, express in arm0 base rotation.
No q8, future root/object pose or force is used. This deliberately excludes
candidate wrist feedback; it tests the retained predictable finger link.
Five anatomical groups are the pinned24points/finger in the original sampler.
For each group concatenate meanxyz and RMSxyz endpoint displacement, fixed
20mm scale:30physical numbers padded32. Arm0 is exactlyzero, signs retained
by mean; RMS alone is unsigned. No learned spatial bottleneck or PCA of action.
Matched Joint input is corresponding fingerq12 candidate-minus-zero/.32,
padded32. All fields come from identical nominal/predicted q, never arm onehot.

## Seven matched heads and action-centering

All heads commonH+32action→64tanh→32tanh→26, initialization245,
batch64/source-window schedule246,300AdamW.001/wd.01updates,clip2. Same
parameter count, architecture, target scales and update/sample schedule.

1. NomJointPlain: ordinary residual f(H,a).
2. NomJointCentered: f(H,a)−mean15 f(H,c).
3. NomFingerPlain: full intrinsic per-finger mean/RMS ordinary residual.
4. NomFingerCentered: same input with candidate-centering.
5. PredJointCentered: causal execution finger-joint input.
6. PredFingerCentered: causal execution intrinsic finger flow.
7. PredFingerShuffled: centered forecast flow with intended arm permuted within
   wave/motion/phase, separatelytrain/test,251, on recipient candidate features.

Prediction mu(H)+scale*residual. Centering is an explicit uniform15candidate
convention, matching this randomized arm grid; it is not identified paired
counterfactual supervision or a double-robust estimate. Plain vs centered
isolates the objective at fixed representation. Joint vs finger isolates
representation at fixed objective; nominal vs forecast isolates execution input.
Train labels are factual E/I only. No future E/I grid exists.

## Fixed gates and scientific boundary

Report all E/I MSE, source residual fit, paired environment bootstrap251/2000,
all15same-state consequence predictions and14signed arm-minus-zero vectors.
Compare candidate contrasts to the same exposed prior adjusted marginal GT
vectors, with raw/arm-centered/+minus scores. This is exploratory calibration,
not a fresh independent GT test. Baseline State/frozen directJoint/Arm and
previous predicted-flow controls are reused and labeled prior fits; only the
seven new heads form the matched factorial.

A for PredFingerCentered: I gain≥5% vsState,lower95>0,≥3% over trainedshuffle,
and frozen test-action shuffle penalty≥3%. B: I contrast corr≥.5,sign≥.65,
gainvszero≥.10,arm-centered gain>0. Geometry-specific:≥3% I gain vs matched
PredJointCentered and frozen directJoint, bothlower95>0. PROMISING for this
causal geometry path only A+B+specific; otherwise fixedfit UNPROMISING.
Report factorial separately; a joint or plain branch win is a local signal,
not geometry utility. Uncertainty and finite single-fit budget remain explicit.

Positive permits a separately frozen strict consequence OOF/task-information
experiment. It would require nested nuisance fits respecting each consequence
source fold; current full-test-only screen does not deliver that OOF. Negative
stops this head/objective/input contract without epoch/seed/dose changes.
Independent review on anomalies/closing decision. No task model, selector or
PPO in this probe; global matched trained-policy Cm utility remains OPEN.

## Artifacts and deferred evidence

Bind original dataset, source spatial and corrected forecast manifests/weights,
code/URDF/mesh hashes. Save seven heads, common source stats, typed action
summaries, training curves, raw predictions/candidates and immutable nuisance
references under outputs/cm-interaction-oracle/relative-finger-innovation-s251.
Do not duplicate prior checkpoints. Preserve failed/smoke runs; smoke is
engineering-only UNCLEAR. New split/dose, nonlinear anatomy encoders, full
temporal geometry, nested OOF and matched policy validation are deferred until
results can change the next decision. None is waived as a completion requirement.

## Completed matched results

Main `relative-finger-innovation-s251`, code8809fee, GPU6:9.151s model/run
stage,9.297s complete manifest, peak318.26MiB. Seven15034-parameter heads
share the exact initialization hash and300 source-window updates. Reused
four State nuisance fits, with full/fold0/1/2 saved-weight replay errors0;
no new simulation, downstream model or policy fitting. Smoke3.77s is
engineering-only UNCLEAR and does not count as scientific evidence.

| Head | E test MSE | I test MSE |
| --- | ---: | ---: |
| NomJointPlain | .520297 | .928197 |
| NomJointCentered | .522982 | .921983 |
| NomFingerPlain | .521021 | .939210 |
| NomFingerCentered | .520646 | .902443 |
| PredJointCentered | .522486 | .907727 |
| PredFingerCentered | .517559 | .912367 |
| PredFingerShuffled | .526123 | .949073 |
| Frozen State (prior fit) | .501971 | .883696 |
| Frozen direct Joint (prior fit) | .498742 | .867037 |
| Frozen predicted Joint (prior fit) | .494638 | .861993 |
| Frozen predicted Surface (prior fit) | .506201 | .897249 |

PredFingerCentered I gain vsState−3.24%(95%CI−5.94..−.62%), vs matched
PredJointCentered−.51%(−4.25..+3.49%). Gain vs trained shuffle+3.87%
(−1.27..+8.89%); frozen test-action shuffle increases I error2.75%, with
the paired interval crossing zero. Thus A and geometry-specific gates fail.
NomFinger centering improves I3.91%(−.99..+7.69%) over plain, while Joint
centering improves.67%(−4.27..+4.47%); neither is a supported win. Their
paired factorial isolates these local interventions, not all geometry models.

PredFingerCentered I raw contrast corr.1565, signed agreement56.57%,
amplitude ratio.2705 and gainvszero1.58%; arm-centered gain−6.81%.
Plus-minus subset corr.3708/sign64.56%/gain15.50% is exploratory and
does not rescue B. NomFingerPlain rawcorr.3307/sign57.71%/gain9.09%
also fails the registered contrast gate. Larger contrast amplitude than the
prior PredSurface.0235 is action sensitivity, not accurate physical prediction
or unique Cm task utility. All three registered gates are false.

Action-centering is actually enforced: test candidate-mean normalized residual
RMS5.38e−8..9.74e−8, versus plain.2422/.2500. Source normalized residual
MSE remains.7265(PredJointCentered),.7614(PredFingerCentered),.7670(shuffle).
These finite300-update fits do not establish convergence or an impossibility
of learning a physical law. Uniform15-centering is a convention; factual
randomized windows do not supply paired state counterfactuals, and a state
predictor is not automatically the exact conditional randomized-arm mean.

Root GPU audit3.13s regenerates common H156, all four action matrices,
source-only extra norms, four nuisance predictions and seven saved heads,
all test candidates and the frozen shuffle. Every replay/metric/raw-contrast
error is0. Candidate signed vectors and standalone figure accompany the run.
62Task contract tests pass. Independent closing review is recorded below.

Independent review reproduces source-fold coverage, source-only common stats,
recipient within-stratum shuffle, initialization and seven small-head algebra.
CPU/GPU maximum raw difference1.38e−5 is normalized1.36e−6; this device
matmul/tanh precision difference does not alter any gate. Root compare finds
metric/raw-contrast errors0 and bootstrap error≤2.39e−7. Root GPU input/FK/
spatial-weight replay remains exact0. No implementation defect affecting this
negative was found. The independent check is CPU saved-algebra/statistics.36s,
avoiding GPU startup and repeating root FK; no fit or simulation.

Saved nuisance diagnostic: source State OOF I1.0189 versus full-fit.7526,
their difference MSE.1171; test both.8837. This source-OOF/full-test stacking
shift and inability of centered residuals to repair common conditional bias
are explicit limitations, not demonstrated causes. Global standardized I
residual mean RMS.0327source/.0755test is small but does not exclude local
bias. Root accepts stopping only this fixed contract. Reviewmd/json and
`review_provenance.json` bind the independent evidence to the immutable run.
Combined main+smoke about24MiB, below40MiB experiment cap.

## Root decision and limits

Stop this fixed finger mean/RMS representation and centered residual fit;
do not repeat its epochs, seeds or action dose. Full anatomical summaries
removed the earlier local pooling interface, yet this matched screen did not
qualify. The result does not distinguish loss of detailed contact geometry,
imperfect state nuisance, finite fitting or conditional response heterogeneity.
No claim that explicit finger IDs or every spatial/point-flow method has failed.

Any next Probe must distinguish one of those mechanisms with a decision that
changes the research route. Strict nested consequence OOF and matched trained
policy Cm-on/off remain deferred requirements, not delivered evidence. The
global North-star utility is OPEN. This is a full-test exploratory screen on
the already exposed854-window grid; adjusted marginal GT contrasts are noisy
and share this dataset. No unseen-dose/continuous generalization or causal
sufficiency claim follows from the action decoding or this factorial.

Main folder preserves `manifest.json`, `fit_records.json`, `diagnostic.pt`,
`result.json`, `engineering_replay.json`, `per_arm_contrasts.json` and
`relative_finger_innovation.png`. Diagnostic binds H/actions/source norms,
seven weights, predictions/candidates and within-stratum shuffle indices.
