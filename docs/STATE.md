# Standalone Contact-Response Research State

Updated: 2026-10-01. Worktree `Ref2Dex-agent-contact-response`, branch
`agent/contact-response-cm`. This branch is an independent research track.
The original worktree remains on `agent/paired-evaluator`; its existing changes
were preserved. Its checkpoints, prior results and motion inputs are read-only.

## Current objective

Under the user's authorization to autonomously extract ideas, experiment and
write a journal-level paper, investigate whether reliably measurable physical
action effects can be learned and ultimately improve dexterous decisions.
Existing literature means generic particle world models, value awareness,
action recovery and controllability representations cannot be claimed as new.

## Confirmed observations

- Physical pulse/repeat experiment completed16/16runs,384windows across four
  actor/evaluation panels and three airplane motions; code466e80d. Five/ten-step
  position contrasts7.15/18.83mm exceed observed zero-repeat contrasts0.10/0.44mm.
  Vertical effects have mixed signs; the full predeclared screen is UNPROMISING.
  This distinguishes physical response from beneficial lift, not policy utility.
- A separate documented vector-prediction screen completed9/9fits with fixed
  three optimization seeds,1000updates each; code44f22ea. Differential contrast
  errors10.11/26.99mm do not beat factual or global-fit-mean controls. Sign
  agreement32.6%/31.1% fails its75%gate. This implementation is UNPROMISING.
- All learned test panels were part of the earlier physical pilot. Results are
  exploratory reused-data evidence, not independent Validation.
- Current-geometry audit:318/384old triggers are within20mm. Actor287/motion1
  has no near examples in either old test panel (gaps55–105mm), exposing a
  systematic geometry shift. This does not prove the cause of learning failure.
- Archived one-step factorization screen (`bbd5ba4`) completed53.64s. Predicted
  hand-motion effect RMSE15.07mm versus native-command16.44mm near the object;
  its8.33%gain fails the fixed10%gate, UNPROMISING. The separately reported
  available PD-target-motion variant scores14.60mm (11.18%gain), a secondary
  hypothesis requiring fresh causal testing. Future-hand oracle is privileged.
-30factorization RMSE metrics independently recomputed from saved predictions;
  feature/acquisition tests pass. No formal scientific conclusion is upgraded.
- Exact-null inverse-recovery screen (`33d1085`) completed12fits;32896states
  pass exact PD-target invariance. Full inverse null-MSE1.0019does not beat the
  population floor1and null response is below factual-only in everyseed.
  UNPROMISING for the proposed interference mechanism; abandon this explanation.
  Hard command canonicalization enforces null invariance but is not a new claim.

## Current decision

Freeze these exact designs and preserve their failed gates. Do not increase
updates, choose a better seed, or start PPO from a failed predictor. The frozen
nominal/command/state models are now tested on fresh geometry-conditioned
plus/minus wrist interventions in all three world axes. No refitting or model
selection is allowed in this transfer screen. This is a Decision Probe, not
policy Validation or new-object generalization.

Fresh `P-20261001-fresh-causal-transfer-r1` completed32/32physical arms and384/384
geometry-conditioned windows, then FAILED the strict pre-intervention match:
seven arms in panelt286_s491 exceed joint/orientation limits, including zero_b.
Do not widen the tolerance, exclude arms, or report a model winner. Scientific
transfer label UNCLEAR; frozen nominal model remains untested causally. All36
native output files and102input hashes independently verified after closure.

Randomized trial `P-20261001-randomized-effect-risk-r2`, code510a00e, COMPLETED:
3072/3072new windows, four768-environment panels,137.84s/8.29MiB. The r1device
alias failure (35.85s) is preserved. All assigned propensities and action changes
are audited. Nominal-minus-command risk difference+0.1223mm², upper95%=1.0126;
nominal-minus-zero-0.3261mm², upper95%=1.1242. All three gates fail, UNPROMISING.
This is an identified risk DIFFERENCE, not absolute effect RMSE. No subgroup
or seed replaces the failed gate. Stop direct control use of these archived
factual predictors; their causal advantage is not supported by the new trial.

Direct randomized learner (`2075314`) COMPLETED:3072fit and3072newtest windows,
initial seeds494/495,IIDseven-arm assignment. All five gates fail, UNPROMISING:
direct-minus-factual risk+23.9773mm²,upper95%=44.2011; every seed worse.
Compute-matched/global/zero controls also not beaten. Six nuisance-fold scales,
16fit model/data files,135test input hashes, predictions and risk/bootstrap arrays
pass independent CPU/NumPy audits. Fit/test30.31/134.49s;3.41/11.89MiB. Stop
this exact learner; no further updates or selected subgroups.

POST-HOC candidate evidence: the simpler factual1000-update control has
factual-minus-zero/global risk-3.4359/-3.1576mm²,upper95%-1.1063/-1.2216,
negative in each fixed seed. This does not upgrade the failed direct primary
gate. No robust benefit of3000versus1000updates appears. The baseline is not
novel and has not demonstrated task benefit. Next frozen Decision Probe tests
equal-weight factual selection against actor/random/global controls on new
seeds496/497, full first-episode retained success and drops, bounded corrections.
No PPO yet.13focused learner checks and6task-controller checks pass.
Task Probe `P-20261001-randomized-task-selection-r1` ACTIVE, code c0ec955,
four768-env panels on independently admitted GPU5;GPU4occupied by another task.
Models/fit-global scores/source SHA frozen before physics. Budget3600s/2GiB;
no result or policy benefit yet. Current process state is in its run manifest.
Original-worktree data remain read-only and all outputs remain isolated here.

## Paper and evidence

- [Results](research/20261001-contact-response-results.md)
- [Literature/novelty screen](research/20261001-contact-response-literature.md)
- [Vector-learning decision](decisions/D-20261001-vector-response-learning.md)
- [Factorization results](research/20261001-actuation-effect-results.md)
- [Fresh causal-transfer decision](decisions/D-20261001-fresh-causal-transfer.md)
- [Fresh matching failure](research/20261001-fresh-causal-transfer-results.md)
- [Randomized estimand decision](decisions/D-20261001-randomized-effect-risk.md)
- [Randomized trial results](research/20261001-randomized-effect-risk-results.md)
- [Direct randomized results](research/20261001-direct-randomized-response-results.md)
- [Task-selection decision](decisions/D-20261001-randomized-task-selection.md)
- [Next task Probe](experiments/probes/P-20261001-randomized-task-selection.md)
- [Current manuscript](../paper/manuscript-v3.tex) and [PDF review copy](../paper/manuscript-v3.pdf)

The manuscript reports actual methods and negative pilot results. Journal
readiness is NOT READY: distinctive method, task-level matched policy benefit,
independent object/task/hand evidence and hardware evidence remain missing.
Original North-star scientific claims are not upgraded by this branch's pilots.
Resource and external-data protection boundaries remain in CAMPAIGN.md.
