---
schema: ref2dex.probe.v2
probe_id: P-20261005-execution-geometry
experiment_id: P-20261005-execution-geometry
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: ff4d8ed
claim_id: C3
hypothesis_family: HF-execution-geometry
probe_index_in_family: 1
seed_pool: probe
seeds: [245, 246, 247]
decision_changed_if_positive: qualify predicted execution flow for strict consequence OOF and independent candidate evaluation
decision_changed_if_negative: distinguish execution approximation from spatial information compression without repeating fit budgets
status: UNPROMISING
run_id: execution-geometry-continuous-fix-s247
---

# Is nominal execution the missing geometric action contract?

Result: UNPROMISING for the corrected fixed endpoint spatial contract. Original forecast-derived negatives are invalid.
Decision: Stop repeated fitting of this total-endpoint spatial contract; retain predictable finger execution and separate common wrist motion in the next physical action design.

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

## Corrected results and root attribution

Original main c2baa30:35.59s GPU6; smoke3.78s oneupdate is engineering-only
UNCLEAR. Corrected run ff4d8ed:21.73s, peak302MiB GPU6; all8 execution weights
and two realized weights reused, only three affected300update fits rerun.
Combined output154MiB. Root audit2.46s reproduces full/OOF q, all5spatial
weights (two referenced original), every15candidate and testshuffle exactly0;
normal-equation relative residual≤6.66e−16. Source four controls and two
realized oracles replay exactly0 before affected training. No future inputs,
new environments, downstream scorer or policy. Regression and57Task tests pass.

| Endpoint input | Scaled q18 MSE | Finger12 MSE | Surface XYZ-component RMSE(mm) |
| --- | ---: | ---: | ---: |
| H | 0.627921 | 0.277613 | 33.625 |
| Ha | 0.479347 | 0.056039 | 34.089 |
| Nominal | 0.609007 | 0.184571 | 32.301 |
| Persistence | 1.448021 | 0.255369 | 50.993 |

Ha finger gain vsH79.81% (95%CI75.70–83.48), vsNominal69.64%
(CI56.15–77.34). Surface squared-error gain vsNominal−11.37%
(CI−132.24–78.04): no execution gate, wide environment uncertainty.
This does not negate finger predictability. Primary RMSE averages XYZ
components; point Euclidean RMSE is sqrt(3) times the listed metric.

| Consequence input | E MSE | I MSE |
| --- | ---: | ---: |
| State | 0.501971 | 0.883696 |
| Nominal | 0.503900 | 0.889131 |
| Arm | 0.496346 | 0.870167 |
| Joint | 0.498742 | 0.867037 |
| RealizedSurface | 0.484325 | 0.891265 |
| RealizedJoint | 0.459550 | 0.841545 |
| PredictedSurface | 0.506201 | 0.897249 |
| PredictedSurface_test_shuffled | 0.507504 | 0.899950 |
| PredictedJoint | 0.494638 | 0.861993 |
| PredictedSurfaceShuffled | 0.504117 | 0.891593 |

RealizedSurface I gain vsNominal−.24% (CI−3.31–2.83), E+3.88%
(CI1.19–6.97); RealizedJoint I+4.77% vsState (CI1.17–8.06),
E+8.45% (CI4.86–12.05). Oracles are post-treatment and cannot qualify
causal deployment. The richer measured execution is useful in a joint channel
at this fixed budget, but its endpoint spatial encoding does not transfer the
same gain. Compression, capacity/optimization, temporal geometry and omitted
fields remain unresolved; no unique causal attribution.

Corrected PredictedSurface I worsens vsState1.53% (CI−.58–4.00),
vsPredictedJoint4.09% (CI1.53–6.77); trainedshuffle is better by.63% point
estimate. Frozen test-action shuffle penalty only.301% (CI crosses0).
I contrast corr.1523, sign50.86%, amplitude.0235; gainvszero.564%,
centered1.588%, +/-pair2.185%. Allfive original gates false. Limited
UNPROMISING applies to this fixed spatial endpoint contract; it does not
refute geometric actions, all OI-CmV2 models or global Cm utility.

### No-fit FK decomposition (post-result, not independent confirmation)

| Endpoint construction | XYZ-component RMSE(mm) | Median window(mm) | Top5 squared-error share |
| --- | ---: | ---: | ---: |
| Corrected forecast | 34.089 | 11.292 | 83.45% |
| Predicted wrist + actual fingers (oracle) | 33.937 | 11.276 | 83.44% |
| Actual wrist + predicted fingers (oracle) | 2.127 | 1.633 | 26.92% |
| Nominal wrist + predicted fingers (causal) | 32.146 | 27.331 | 18.90% |
| Current wrist + predicted fingers (causal) | 50.921 | 41.586 | 16.64% |

Actual wrist replacement lowers squared error99.61% (CI98.17–99.79),
actual finger replacement only.89% (CI crosses0). Root interprets this as
wrist-dominated factual endpoint error, not unpredictable finger action.
Causal nominal/current wrist substitutions do not recover oracle accuracy;
nominal wrist hybrid improvement vsPredicted11.07% has wide CI crossing0.
No rows removed and no consequence models fit to those hybrids. Thus simple
wrist substitution is not enough to qualify further endpoint spatial fitting.
First decomposition invocation failed on scalar geometry `.to` before FK;
2f9ef65 fixes tensor/scalar dispatch, original log retained. Successful hybrid
and causal reports bind diagnostic/code hashes separately.

Root Decision: end this total-endpoint fit family at the declared budget.
Preserve continuous action→finger execution as a positive local Probe signal,
withhold all downstream/selector/PPO. The next smallest useful physical design
should explicitly separate shared wrist/base evolution from finger-induced
relative geometry or inspect information retained by the spatial bottleneck;
do not increase epochs/width or discard outliers to pass the exposed split.
Common-frame/relative action contracts must still use only pre-action input,
recipient candidate regeneration and independent source environment OOF.
Future strict consequence OOF, new split, realized temporal path and matched
trained-policy comparisons remain deferred until causal action gates qualify.

Artifacts: original and repaired immutable folders under
`outputs/cm-interaction-oracle/`; corrected `manifest.json`, `result.json`,
`diagnostic.pt`, `fit_records.json`, `engineering_replay.json`,
`per_arm_contrasts.json`, `execution_diagnostics.md`, standalone
`execution_geometry.png`, `endpoint_decomposition.json` and
`causal_endpoint_contracts.json`. Slim repaired diagnostic references original
weights/geometry instead of duplicating them. Raw per-arm driver travels and
all14signed consequence vectors are retained; do not treat travel as adjusted
causal arm-minus-zero differences.


## Independent review accepted by root

Reviewer reconstructed all candidate forecasts from the eight saved source
ridge weights, source normalizers and environment folds; continuous projection
and unaffected controls/oracles agree exactly. Independent E/I metrics,
bootstrap and candidate contrasts agree; root metric comparison max0.
Additional no-fit GPU inference1.42s, saved PredictedSurface replay0.
99.30% windows have local edges, test mean22.81active object points;
recipient action changes KNN on99.50% treatment rows and fused representation.
The spatial action pathway is active, with factual-vs-zero I response RMS.01885,
so weak contrast is not missing action input. These are implementation checks,
not evidence of policy value. Root accepted after actual-FK, projection, OOF,
metric and hash checks; review does not remove finite-fit/generalization limits.

Corrected folder contains independent_review.md/json and review_provenance.json
binding their hashes and corrected diagnostic. Original `probe_execution_geometry.py`
is a frozen historical reproduction entry with the invalid all18 projection;
use repaired entry for this design. Do not launch new work via the historical
runner. Original code hashes are intentionally preserved for traceability.
