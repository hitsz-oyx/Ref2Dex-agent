# P-20260924-multiaxis-h10-effects

date: 2026-09-24
branch: `agent/cm-multiaxis-long-horizon`
classification: Decision

## Question and decision

Does a real executed action on wrist x or y, beyond the studied wrist z,
cause a repeatable 10-step object-motion or contact effect in the
self-trained policy's contact states? If yes, collecting multiaxis data
and training a new action-conditioned Cm is worth the next investment.
If both new axes have negligible effects, do not expand the Cm yet;
return to representation/longer-horizon task alignment instead.

This is an average randomized intervention effect on reached states,
not a same-state counterfactual or evidence of Cm policy utility.

## Cheapest discriminating protocol

Keep the pinned self-trained e260 actor and corrected single-motion
input. At global steps 50..200 with stride 10, in actual contact and
not resetting, randomly assign each eligible environment to one of six
roughly balanced cells: wrist x/y/z action ±0.1 (one control step only).
Require all three pre-action coordinates to be within ±0.9, so no dose
is clipped. Then resume the unmodified actor and record one-step and
10-step object state, 10-step contact fraction/final contact and
survival. Run seed161 first, then seed162 if assignment and followup
contracts pass. Each run uses 64 environments and one idle GPU.

For each axis, compute step-stratified plus-minus effects on the
matching world object-translation component (mm), 10-step contact
fraction (pp), and first-step object motion. Cluster uncertainty by
environment, preserving the treatment step strata. Continue to Cm
training only if at least one non-z axis has a same-sign 10-step
object displacement ≥5 mm or contact difference ≥3 pp in both seeds,
with pooled environment-cluster 95% CI excluding zero. Otherwise
classify `UNPROMISING` for this sampled multiaxis intervention (or
`UNCLEAR` if assignment/sample size is insufficient) and stop.

## Budget and safety

At most two sequential GPU runs, 64 env each, 1 GPU/run, <60 min total,
<400 MB additional output. Stop for input/code drift, GPU occupancy,
unbalanced or clipped assignment, invalid followup, non-finite state,
or budget breach. No new policy training in this first decision Probe.

## Result

Status: PENDING

Evidence: pending

## Decision update

pending
