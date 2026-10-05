---
schema: ref2dex.probe.v2
probe_id: P-20261005-oracle-flow-task
experiment_id: P-20261005-oracle-flow-task
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: pending
claim_id: C3
hypothesis_family: HF-oracle-flow-task
probe_index_in_family: 1
seed_pool: probe
seeds: [255, 256, 259, 260, 261]
decision_changed_if_positive: retain the OOF oracle flow consequence task chain and assess its contribution beyond direct flow
decision_changed_if_negative: distinguish lost consequence task information from weak oracle task denominator without planner or execution fitting
status: UNCLEAR
run_id: oracle-flow-task-s259
---

# Ref12: oracle hand flow → OOF predicted E/I → task Y

Result: UNCLEAR: protocol frozen; runs pending.
Decision: Close the factual oracle prediction chain on existing data; paired-data and execution work remain paused.

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
