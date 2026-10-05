---
schema: ref2dex.probe.v2
probe_id: P-20261005-relative-finger-innovation
experiment_id: P-20261005-relative-finger-innovation
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: pending
claim_id: C3
hypothesis_family: HF-relative-finger-innovation
probe_index_in_family: 1
seed_pool: probe
seeds: [245, 246, 251]
decision_changed_if_positive: qualify global anatomical finger flow and action-centered learning for strict consequence OOF
decision_changed_if_negative: distinguish the fixed geometry representation from state-residual learning and avoid repeated fitting
status: UNCLEAR
run_id: relative-finger-innovation-s251
---

# Can full finger flow learn action innovations over an OOF state forecast?

Result: UNCLEAR: protocol frozen, run pending.
Decision: Test anatomical global flow and action-centered residual jointly with a matched factorial.

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
