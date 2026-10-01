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
-14focused tests pass; GPU4 is released. Physical and learning runs cost481.9s
  and17.2s, respectively. Experiment artifacts occupy about70MB.

## Current decision

Freeze these two exact designs and preserve their failed gates. Do not increase
updates, choose a better seed, or start PPO from this failed predictor. The next
useful question concerns verified interacting geometry and broader contact/action
coverage on a newly frozen independent test set. No further run is active.

## Paper and evidence

- [Results](research/20261001-contact-response-results.md)
- [Literature/novelty screen](research/20261001-contact-response-literature.md)
- [Vector-learning decision](decisions/D-20261001-vector-response-learning.md)
- [Working manuscript](../paper/manuscript.tex) and [PDF review copy](../paper/manuscript.pdf)

The manuscript reports actual methods and negative pilot results. Journal
readiness is NOT READY: distinctive method, task-level matched policy benefit,
independent object/task/hand evidence and hardware evidence remain missing.
Original North-star scientific claims are not upgraded by this branch's pilots.
Resource and external-data protection boundaries remain in CAMPAIGN.md.
