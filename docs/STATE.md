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

Next: prospective randomized pulse trial `P-20261001-randomized-effect-risk`.
New initial seeds492/493, four768-environment batches, frozen models. Known
assignment probabilities identify conditional-effect RISK DIFFERENCES without
asserting individually cloned solver states or absolute effect RMSE. Analytical
identity checks pass under uniform and unequal propensities. Three additional
tests pass (25relevant checks total across this stage). GPU5 is released.
Before resuming, inspect the new experiment's run manifest/process for an active
run; do not relaunch from transient waiting. All outputs stay in this worktree.

## Paper and evidence

- [Results](research/20261001-contact-response-results.md)
- [Literature/novelty screen](research/20261001-contact-response-literature.md)
- [Vector-learning decision](decisions/D-20261001-vector-response-learning.md)
- [Factorization results](research/20261001-actuation-effect-results.md)
- [Fresh causal-transfer decision](decisions/D-20261001-fresh-causal-transfer.md)
- [Fresh matching failure](research/20261001-fresh-causal-transfer-results.md)
- [Randomized estimand decision](decisions/D-20261001-randomized-effect-risk.md)
- [Current manuscript](../paper/manuscript-v2.tex) and [PDF review copy](../paper/manuscript-v2.pdf)

The manuscript reports actual methods and negative pilot results. Journal
readiness is NOT READY: distinctive method, task-level matched policy benefit,
independent object/task/hand evidence and hardware evidence remain missing.
Original North-star scientific claims are not upgraded by this branch's pilots.
Resource and external-data protection boundaries remain in CAMPAIGN.md.
