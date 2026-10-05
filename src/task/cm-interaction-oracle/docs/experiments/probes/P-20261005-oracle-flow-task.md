---
schema: ref2dex.probe.v2
probe_id: P-20261005-oracle-flow-task
experiment_id: P-20261005-oracle-flow-task
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: c2e55b22d6c0b4a75e162e8ea7f1c2ed5da904ce
claim_id: C3
hypothesis_family: HF-oracle-flow-task
probe_index_in_family: 1
seed_pool: probe
seeds: [255, 256, 259, 260, 261]
decision_changed_if_positive: retain the OOF oracle flow consequence task chain and assess its contribution beyond direct flow
decision_changed_if_negative: distinguish lost consequence task information from weak oracle task denominator without planner or execution fitting
status: UNPROMISING
run_id: oracle-flow-task-s259
---

# Ref12: oracle hand flow → OOF predicted E/I → task Y

Result: UNPROMISING for the registered fixed-fit chain; direct oracle flow task information is PROMISING, predicted E/I increment remains uncertain.
Decision: Preserve direct flow task information; stop this fixed chain fit without planner, paired data, execution or epoch/seed extensions.

## Root Decision Note / Decision experiment

User directs ref12 after deferring paired data. MissionB needs task-relevant
physical predictions; ref11 Chunk E gained7.42%, I8.07% with CIcrossingzero,
while ref8 GT E/I prognosis gained46.25% under its different H/fit contract.
Question: does a cross-fitted raw actual-flow consequence model transmit task
information, beyond shared current state and against a direct flow readout?
Cheapest test: same854windows,688/166split (125/31environments), E12/I14 and
ref8 Y8 unchanged. Primary flow is ref11's predeclared two chunks, not chosen
between endpoint/temporal after test. Keep original branch, paired collection
and native execution forecasting paused. No simulator/control/planner/PPO run.

Reuse immutable ref11 full State/Chunk weights for OUTER TEST only. Six new
consequence fits (3folds × State/Chunk) and six matched task heads. GPU6 ifidle;
main≤120s, smoke≤120s, replay≤120s; eachrun<30MiB, combinednewartifacts≤90MiB.
Stop on hash/split/frame drift, fold leakage, nonfinite, invalidpointscale,
resource conflict or storage cap. Positive qualifies further representation-chain
work, negative stops this fixed-fit chain; no claim or permission change.
No external authorization needed. Multi-seed Validation deferred.

## Strict source OOF and inherited physical contracts

DatasetSHA138b99b21e5567616dbdb6556179cc87dc8b42b578cad7e127dc6d5590e17149.
Parent `oracle-hand-flow-s255` commit452d7e5, all hashes bound in manifest.
Reconstruct current-frame measuredq0/q4/q8 raw Chunk720 and currentH156;
verify live tips/base, original full H/geometry/flow/labels/weight predictions.
No PD, arm, futureobject/force or execution forecast in H/flow. Only actual
future HAND motion is deliberately allowed as a post-treatment oracle input.

Three motion-stratified environment folds seed261 exclude outer-test entirely.
Each fold fits H normalization/historyPCA32/currentgeometryPCA16 and E/I label
mean/scale ONLY on its fit environments. Consequence models match ref11:
876→64tanh→32tanh→26, seed255/init, batch64schedule256,300AdamWupdates,
lr.001/decay.01/clip2. State actionblank, Chunk full720/raw20mmscale. ALLsource
rows use exactlyone held-environment prediction; no in-sample/full-source
prediction can fill source task features. Test uses frozen FULL-source models.
Fold membership, normalizers, weights and held predictions retained for replay.

Task H is the same source688 current-only preprocessing for all task heads.
Physical slots26 use source688 GT E/I mean/scale, preserving physical-axis
meaning across GT/OOF/full-test variants; targetY uses source688 mean/std.
OOF feature distribution may differ from full-source test distribution; disclose.
Source in-sample consequence errors are diagnostics ONLY, not task inputs.

## Matched downstream models and outcome definition

All task heads902→64tanh→32tanh→8,60136parameters, sameinit259/batch260,
500updates/minibatch64/AdamW.001/decay.01/clip2. Missing flow720 or physical26
slots are exactlyzero; no PCA/latent bottleneck on flow. Models:

- H: shared current state.
- Flow: H + actualtwo-chunkflow.
- PredEI: H + OOFChunk E/I for source, full-sourceChunk E/I for test.
- GT_EI: H + true step8E/I, a task-information oracle.
- Flow_PredEI: H + actualflow + predictedChunk E/I.
- StatePredEI: H + OOF/full-source State E/I, controls learned state transforms.

Y8 uses inherited contact/held/combinedfailure16/32 and late heightfailure/
height-heldfraction32. ALL Y windows startstep9, after the step8mediator.
PRIMARY axes3/6/7: contact32, heightfailure32late, heightheldfraction32late;
source-standardized MSE primary, all8/per-head, physicalfailureAUC reported.
These are bounded continuation proxies, not final episode success or a policy.
No new task label, future-active test filtering or candidate-label transplant.

Frozen TEST flow shuffles keep recipient H and replace only its rawflow within
parent partition/wave/motion/phase257. Propagate shuffledflow through frozen
Chunk model before evaluating PredEI/hybrid; do not transplant a donor's E/I
(which would also change H). Flow/hybrid share the SAME shuffledflow. These
are sensitivity controls, never attainable counterfactual plans.

## Frozen readouts and decision gates

Report primary/all8/physical per-env-bootstrap2000seed261 gains. Primary R:
`(L(H)-L(PredEI))/(L(H)-L(GT_EI))`, with paired-cluster CI and valid positive
oracle-denominator bootstrap fraction. Weak oracle denominator invalidates
strong R interpretation. No cross-run denominator from oldref8 scores.

Fixed chain PROMISING only when ALL primary gates pass: GT_EI vsH≥10% with
lower95>0; PredEI vsH≥5% with lower95>0; PredEI vsStatePredEI≥3% with
lower95>0; R≥.50/lower95>0/positive-denominatorfraction≥.95; PredEI improves
≥3% against its frozen testflowshuffle with lower95>0. Otherwise fixedfit
UNPROMISING, with per-link information/uncertainty explicitly retained.

Separate unique-contribution label PROMISING only if PredEI OR hybrid improves
≥3% over directFlow with lower95>0; otherwise UNCLEAR, not equivalence or
proof of no Cm contribution. Also report hybrid-vsPredEI and directFlow-vsH.
No choosing another head/metric/threshold after seeing test. Smoke1update is
engineering-onlyUNCLEAR; source/testing adequacy must not be inferred from loss.

## Evidence boundaries and artifacts

This reuses an already exposed test split and single fit/collection seed.
Actualflow may carry object/controller response. A good factual prognosis
chain cannot establish exogenous desiredflow→outcome, candidate ranking,
attainable controls or final Cm policy utility. User explicitly defers paired
states/collection and their engineering smoke; do not restart that work here.

Outputs `outputs/cm-interaction-oracle/oracle-flow-task-s259/`: manifests,
fold assignments, source-only stats/PCA, sixOOF predictors, sixY heads,
raw/normalized features, all predictions, learningcurves/results and audits.
Do not overwrite parent weights or failed/smoke runs. GPU saved-input/weight/
OOF/statistical replay required; anomalous/negative result gets independent
read-only review before closing this fixed contract. Future evidence includes
multi-seed/newstates/objects, strict nested tuning, convergence, paired contrasts,
prospectiveflow-space planning and matched trained-policy Cm-on/off utility.


## Completed matched task readouts

Main `oracle-flow-task-s259`, commitc2e55b2, completed in11.72s model/result
(manifest11.94s) onphysicalGPU6. All854windows retained,source688/test166,
125/31environments; every source window has exactlyone environment-held
prediction. Sixfold consequence fits59066parameters/300updates; sixtask
heads60136parameters/500updates, identical taskinit/source-batch sequence.
Two full consequence weights reused immutably fromref11 for test features.

| Task features | Primary normalized Y MSE | Physical-failure MSE | Physical-failure AUC | Primary gain vs H [95% env CI] |
| --- | ---: | ---: | ---: | ---: |
| H | 0.639513 | 0.717384 | 0.8311 | baseline |
| H + actual Flow | 0.457881 | 0.531071 | 0.9086 | 28.40% [12.23,41.48] |
| H + predicted E/I | 0.547121 | 0.619280 | 0.8485 | 14.45% [-1.57,29.47] |
| H + GT E/I | 0.454401 | 0.608470 | 0.8592 | 28.95% [10.61,42.74] |
| H + Flow + predicted E/I | 0.436941 | 0.508046 | 0.9075 | 31.68% [15.82,43.76] |
| H + State-predicted E/I | 0.701844 | 0.805572 | 0.8072 | -9.75% [-18.00,-1.78] |
| TrainMean | 0.983617 | 0.999518 | 0.5000 | descriptive |

Directflow physicalfailure gain25.97% [5.23,44.36]; predictedE/I13.68%
[-4.11,30.11]; GT E/I15.18% [-10.26,35.66]. Thus primary task information
is not a claim that every task head/physicalfailure improves robustly.
GT E/I primary gain is positive under this NEW commonH/readout contract;
do not reuse oldref8 46.25% as the current ratio denominator.

R=0.499114 (49.91% retained point gain),95%CI[-0.063906,1.278980], valid
positive denominator bootstrap fraction99.85%. GT task information and
predicted-vsStatePredEI increment gates pass (22.05% [4.30,35.64]). The
PredEI-vsH interval gate, R point/interval gate, and frozen sensitivity
interval gate fail. R's wide CI, not its tiny .50point-threshold shortfall,
is the practical barrier to saying most oracle task information survived.
Overall fixedfit chain UNPROMISING; task-information transmission remains
uncertain, not formally refuted.

PredEI vsdirectFlow gain−19.49% [-37.84,-0.84]: this fitted bottleneck readout
has worse primary prediction than directflow. Hybrid vsdirectFlow gain4.57%
[-1.96,10.41], vsPredEI20.14% [4.45,30.52]. Hybrid has the best point estimate
but no demonstrated UNIQUE increment over directFlow; unique labelUNCLEAR.
A learned deterministic E/I transform of H/flow can change readout inductive
bias; it adds no new external observation. StatePredEI worsens H, so its
contrast alone cannot establish a practically useful mediated increment.

Frozen inputflow sensitivity: directFlow testshuffle MSE0.803285 (actual
improves42.999% [27.92,56.57] vs shuffled), PredEI shuffled0.602701 (9.22%
[-9.77,26.82]), hybrid shuffled0.772907 (43.47% [28.06,56.60]). All use SAME
donorflow with recipientH; predictedphysical is recomputed by frozenChunk,
never transplanted from donor. Sensitivity cannot turn these fields into
feasible candidate plans or same-state ground truth.

## OOF, optimization and implementation checks

Source OOF State E/I MSE0.714181/0.663985, Chunk0.720325/0.615501; test
full-source State0.528949/0.604857, Chunk0.489677/0.556029, reproducingref11.
Source FULL in-sample diagnostic Chunk0.539292/0.390915 is materially more
optimistic than its OOF0.720325/0.615501. These in-sample predictions never
entered source Y features. Source OOF learns on smaller source folds and
uses independently fitted PCA/targetstats; full-source test feature quality
and distribution differ. This is a disclosed cross-fitting limitation,
not evidence of future-label leakage or a unique explanation of losttaskgain.

Task source ALL8 standardized MSE H0.192954/Flow0.079180/PredEI0.139765/
GT0.080904/hybrid0.076092/StatePred0.167052. All final50 minibatch losses
still improve against preceding50 (roughly4–13%); no convergence or equal
optimization sufficiency claim. Equal parameter/update budget does not mean
equal active input dimensions or identical effective inductive bias.

Root GPU audit rebuilds actualgeometry, eachfoldH/PCA/targetstats, inherited
full weights, sixfold weights, OOF mosaic, physicalnormalizers, sixY heads,
three recipient-preserving frozen shuffles and stats: ALLreplay/metric errors0.
Independent checks of lateheightfailure(<2cm) and heightheldfraction(≥3cm)
match. FK/live tip maximum0.02989mm; temporalidentity remains1.91e-6 scaled.
Replay2.13s, plot/export2.64s; no newfits. Smoke1update is UNCLEAR engineering
only, fixed-parent300update weights remain inherited. Main+smoke~23.3MiB.
Artifacts include `engineering_replay.json`, `oracle_flow_task.png`, original
manifest/result/diagnostic/fit_records. All72Task tests passed (7.68s), including sourcefold leakage guards and GPU
FK/index contracts. Scoped repository verification and generated index checks
pass. Independent closing review is archived as `independent_review.md/json` plus
hash-bound `review_provenance.json`. Independently reconstructed all8raw Y
labels, sourcefold stats, inherited/fold/task weights and recipientpreserving
shuffles. CPU saved-head normalizedreplay≤2.17e-6, metrics/bootstrap/R/gates
agree; no defect affecting the scientific result was found. CPU was justified
by small saved-head algebra/statistics; full FK/live reconstruction remained
rootGPU-owned. All reviewerCPU passes within120s. Reviewer-only threshold/
reportpath errors were corrected before report completion; no scientific run
or frozen input/weight was changed.

GPU SVD's frozen PCA projections are approximately orthogonal (maximum
PᵀP−I entry5.66e-4 in doubleprecision); source-only score normalization and
full replay are consistent. This is a numerical approximation of currentH
compression, not flow compression or futurelabel leakage. No projection refit
or gate change was made. Independent review does not certify optimization
sufficiency, generalization or a unique cause of the mixed chain outcome.

## Root closing decision and deferred evidence

Ref12's requested comparisons and sourceOOF implementation are executed.
Retain PROMISING directactualflow→Y and GT E/I→Y information. Do not assert
that the E/I bottleneck preserves substantial stable taskgain, that hybrid
establishes a unique Cm contribution, or that the whole flow/Cm route failed.
Stop this particular exposed-test fit; no besthead/epoch/seed/testsubset rescue.

Any later model change must distinguish consequence task-information loss,
sourceOOF/full-test distribution, and source-only optimization/capacity
adequacy under a separately frozen Decision protocol. Such changes are not
started here. The directflow route is a useful matched baseline for future
representation-chain work, not proof of a deployable flow action policy.
Multi-seed/newstates/objects, paired contrasts and their engineering checks,
execution mapping, flow-space planner and matched trained-policy utility
remain deferred; user directions keep paired work and execution forecasting
paused. North-star Cm policy utility OPEN, final success criterion unchanged.
