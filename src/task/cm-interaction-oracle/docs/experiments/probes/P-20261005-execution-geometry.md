---
schema: ref2dex.probe.v2
probe_id: P-20261005-execution-geometry
experiment_id: P-20261005-execution-geometry
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: c2baa30
claim_id: C3
hypothesis_family: HF-execution-geometry
probe_index_in_family: 1
seed_pool: probe
seeds: [245, 246, 247]
decision_changed_if_positive: qualify predicted execution flow for strict consequence OOF and independent candidate evaluation
decision_changed_if_negative: distinguish execution approximation from spatial information compression without repeating fit budgets
status: UNCLEAR
run_id: execution-geometry-s247
---

# Is nominal execution the missing geometric action contract?

Result: UNCLEAR: original forecast projection invalid; affected-only repair frozen below.
Decision: Compare actual-motion diagnostics and strictly cross-fitted execution predictions before new simulation.

## Motivation and root Decision Note

Continue ref10 physical geometric action toward MissionB. Prior reduced V13
spatial model is numerically stable with effective action input but weak
contrast (I amplitude.066) and no task value. Nominal endpoint differs from
factual execution. Question: can physically credible executed geometry offer
useful E/I information, and can intended action forecast that geometry from
decision-time input? This distinguishes execution modeling/acquisition from
spatial information compression. Cheapest discriminator reuses ref7 all854
windows, exact688/166 environment split and previous source-fold preprocessing.
No new simulator, selector or task fits. Exposed test remains exploratory.

Root chooses eight small exact ridge execution fits (H/Ha × full/threefold),
then five300-update matched spatial fits. Reuse frozen State/Nominal/Arm/Joint
controls with exact weight replay instead of rerunning baselines. No extra
epochs, alpha/seed/width/sampling scans. Positive predicted action gates permit
a separately frozen consequence OOF/candidate experiment; negative identifies
which contract remains weak, not Cm or all spatial models being refuted.
No external authorization needed; global claim/budget unchanged.

Single idle GPU6; main wall cap600s, engineering smoke120s, combined new
outputs≤200MiB. Model/FK/PCA/ridge/inference on GPU; CPU stats and tiny synthetic
integration tests because GPU startup dominates. Stop on nonfinite, hash/input
drift, source control replay>1e−6, FK/live mismatch>0.1mm, fold leakage or resource
conflict. Preserve failures. Independent review required on anomalous results.

## Frozen labels and future-information boundary

Source `outputs/cm-interaction-oracle/spatial-consequence-s245/`, execution
0f2480f, completed. Dataset original SHA138b99b21e5567616dbdb6556179cc87dc8b42b578cad7e127dc6d5590e17149.
Use exact train/test and three source folds, frozen source-only H104+geometry16,
source point identities120hand/64object, state geometry and target scales.
E12/I14 exactlystep8; no target change or window exclusion.

Realized q=measured nativeq atstep8 (q0:3 metres,3:18radians). FK with
decision-time actor root agrees with live step8 true tips max0.02804mm and
hand-base matrix max7.57e−6. Realized surface flow uses this FK endpoint minus
CURRENT sampled hand point, expressed in CURRENT object frame. No future
object pose, force, contact normals or E/I is used to construct geometry.
It is an endpoint segment, not the full executed sweep/path.

RealizedSurface and RealizedJoint are explicitly POST-TREATMENT diagnostic
oracles, not causal action inputs. They include endogenous feedback influenced
by evolving object/contact state. Oracle improvement cannot uniquely prove
nominal execution error caused the previous failure, and cannot justify a
selector. Oracle failure also does not prove executed geometry lacks information
at greater spatial capacity/longer training. No counterfactual GT realized
geometry exists for the unexecuted14candidates.

## Source-only execution model and geometric forecast

Current q18, dq18 and nominal source PD target18 are source-only standardized
and appended to frozen H120. Future nativeq is a target only. Predict
(q8−qcurrent)/units using fixed units[.02m×3,.1rad×3,.32rad×12]. Centeredfloat64
ridgeα32, no products; H action18slots0, Ha candidate native PD perturbation
divided by same fixed units. All15hypothetical candidate forecasts regenerate
action features at the RECIPIENT current state. Native target coupling enters
the intended-action columns, but measured future followers remain separate
execution labels. Source-fold raw-feature normalizers only.

Full train fits forecast outer-test. Three source environment folds forecast
outer-train OOF, with every row exactly once and no environment shared between
source/hold. PredictedSurface/Joint train on these OOF execution forecasts;
test uses full-fit forecasts. No in-sample execution prediction becomes a
learned consequence feature. All18forecast q coordinates clipped to declared
native URDF limits; report clipping fractions. This is a projection convention,
not a tuned correction; actual oracle q is not clipped.

Compute endpoint surface flow by FK(predictedq) with current actor root/current
object frame. Predictor changes no future root/object labels. Compare execution
H/Ha with persistence and nominalPD; report test finger12scaled MSE, all18MSE,
surface endpoint RMSE(mm), paired environment-bootstrap and OOF quality.

## Fixed matched consequence comparison and action contrasts

Reuse exact same V13 width32/4tokens/2cmKNN8, seed245,300updates, batch64,
AdamW.001/wd.01/clip2, same minibatch schedule246,26residual-head architecture.
E baseline0/currentI baseline, same source-train target scale. New fivefits:
RealizedSurface, RealizedJoint, PredictedSurface, PredictedJoint,
PredictedSurfaceShuffled. Every model shares the current/nominal source
geometry PCA120state. Joint oracle/forecast uses18deltaq/physicalunit in32slots
and spatial arm0 nominal baseline, while surface variants use total endpoint
flow and action32slots0. All use equal model capacity/init/budget.

Shuffled model uses permuted INTENDED arm within wave/motion/phase, separately
train/test, seed247, then chooses that candidate from recipient OOF/full-fit
execution forecasts. It never moves measured donor flow. Also frozen
PredictedSurface test-action shuffle. Save15same-state predicted geometry
consequence candidates; GT comparison remains prior rank14 adjusted marginal
E/I vectors, not per-state counterfactual truth. Report centered and +/-pairs
to disclose sharedzero effects. All14raw signed vectors exported.

## Predeclared gates and next decision

Oracle diagnostic: RealizedSurface I gain≥10% over Nominal,lower95>0.
Execution: Ha surface error gain≥20% over Nominal AND finger12gain≥10% over
H,lower95>0. These identify information/predictability links, not policy value.
A: PredictedSurface I gain≥5% vsState,lower95>0; trained-shuffle gain≥3%,
frozen test-action shuffle penalty≥3%. B: I candidate corr≥.5,sign≥.65,
gain-vs-zero≥10%,arm-centered gain>0. Geometry-specific: I gain≥3% over
both frozen intended Arm/Joint AND PredictedJoint. Predicted spatial route
PROMISING only A+B+geometry-specific. Otherwise fixedfit UNPROMISING; an
oracle win alone is not a causal positive. Bootstrap247,2000environment draws
on fixed fitted predictions, no refits. Report E as well as primaryI.

Oracle+execution positive but predicted E/I negative prioritizes information
transfer/representation diagnostics. Oracle positive/forecast negative
prioritizes causal execution model or acquisition. Both oracle and predicted
spatial weak stops this endpoint family’s repeated fitting; RealizedJoint
control distinguishes a spatial-compression limitation. No task/scorer/PPO
until causal predicted action gates qualify. Neither stopping nor finite
budget establishes unlearnability or changes Mission claim.

## Artifacts and deferred evidence

Entry `tools/run/probe_execution_geometry.py`, helper `src/execution_geometry.py`.
New folder `outputs/cm-interaction-oracle/execution-geometry-s247/`; refuse
overwrite. Save manifests/hashes, all8executionridge and5spatialweights,
sourcefold stats, full/OOF15candidateq, raw predictions/contrasts and exact
source-control replay. Retain original per-finger commanded/live matrices;
new execution endpoint errors are different quantities and reported separately.
Future independent split, realized path instead of endpoint, original spatial
capacity/pretraining, strict consequence OOF and matched trained-policy
Cm-on/off remain deferred until they can change a decision.

Engineering FK fixture in projecttmp: first attempt used ambiguous cuda alias
and failed device guard before FK; corrected to cuda:0, unchanged inputs.
This is an engineering invocation issue, not physical negative evidence.

## Implementation defect and affected-only repair Decision Note

Original main `execution-geometry-s247` completed on c2baa30. Preserve its
manifest, labels, weights, predictions and nominal machine status verbatim.
Its forecast-derived spatial negatives are **invalid evidence for the intended
continuous-joint model**. Root introduced an all18 absolute clamp; URDF
joint4/5/6 are continuous, with no lower/upper. Shared native_joint_limits
falls back to ±pi for controller scale, not physical forecast bounds.
Simulator front6 targets are currentq+scaledaction. q8 wrist ranges include
[-4.811,3.162],[-6.075,1.225],[-2.071,4.781]rad. No q8−q0 rotation exceeds
.575rad, so a wrap discontinuity is not the cause. Ha test q18scaled MSE
increased from raw .483786 to 2.307839 after erroneous clipping.
Independent reviewer confirmed this semantic defect. Regression with actual
URDF FK and q=[pi+.3,-pi-.4,pi+.2] failed under old projection and passes
when continuous coordinates retain their unwrapped simulator branch.

Question: do corrected causal forecasts improve executed geometry and transfer
to I? Reuse all8 source/OOF ridge weights; regenerate recipient15forecasts,
project only truly bounded DOFs via URDF type mask, preserving indices3:6
exactly. Reuse two valid realized-oracle fits and four source controls with
exact replay. Only refit PredictedSurface/Joint/Shuffled (same245/246 seeds,
300updates, no budget extension or threshold change). Recompute all original
gates and errors, then saved-weight replay and independent review. No new
experiment identity or seed: this repairs the same design. Fixed forecast
projection code lives separately, leaving original and source hashes intact.

New `run_id: execution-geometry-continuous-fix-s247`, run entry
`tools/run/repair_execution_geometry.py`, helper `src/execution_projection.py`.
GPU6, ≤600s, slim diagnostic references original immutable assets/weights to
keep total smoke+original+repair outputs ≤200MiB. No extra repair smoke fit:
regression test and exact source/oracle replay gate before affected training.
Stop on nonfinite, changed hashes, leakage, replay drift or resource conflict.
Success qualifies the original next decision; failure stops repeated endpoint
spatial fitting at this capacity. No selector/PPO, claim change or external
authorization. The two original realized oracles and all8 raw ridge fits are
unaffected; their evidence scope remains post-treatment diagnostic/execution.

## Post-result no-fit execution decomposition protocol

Decision: distinguish common wrist prediction error from finger response error
before designing further geometric input. Frozen corrected full-test forecasts
are compared with two post-treatment FK hybrids: predicted wrist6 + actual
finger12, and actual wrist6 + predicted finger12. No fit, excluded rows or
modified gates. Large improvement only after actual wrist would prioritize a
separate common wrist versus finger action contract; improvement only after
actual fingers would prioritize finger execution. Neither hybrid is deployable
input or causal utility evidence. GPU6≤30s; small JSON only. Also report top1/5
error shares and median without dropping outliers. Primary surface RMSE is
sqrt(mean squared XYZ component); show sqrt(3) Euclidean point RMSE separately.

Decomposition root follow-up Decision: actual wrist replacement reduced
endpoint RMSE34.09→2.127mm; actual fingers only34.09→33.94mm. Errors are
concentrated (top5 83.45%), but no rows excluded. Next cheapest discriminator
is no-fit fully decision-time alternatives: source nominal PD wrist + predicted
fingers, or current wrist + predicted fingers. Same frozen full-test forecast
and all166rows; actual q8 enters only metrics. GPU6≤30s and small JSON. Save
separate causal_endpoint_contracts.json; keep original hybrid report unchanged.
These endpoint errors alone do not qualify consequence or policy training.
