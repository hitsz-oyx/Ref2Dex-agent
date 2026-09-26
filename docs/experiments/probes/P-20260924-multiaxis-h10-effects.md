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

Status: `PROMISING` for multiaxis executed-action information, not Cm
policy utility. Code: `1c9e2a6` (collector), `8920e59` (analysis).

Both 64-env runs `agent_multiaxis_h10_s161_n64` and
`agent_multiaxis_h10_s162_n64` completed with 16 intervention steps,
respectively 817 and 757 contact-eligible treated rows. The exact
±0.1 wrist dose, six-cell per-step balance and ten-step followup passed
contract checks. Inputs/checkpoints and their SHA256s, commands and
logs are in each run's `run_manifest.json` and `collect.log`.
Pooled analysis: `outputs/CmResidual/agent_multiaxis_h10_s161162_analysis/`
(`report.json`, `run_manifest.json`; 500 environment-cluster resamples).

| Axis / outcome, plus-minus | seed161 | seed162 | pooled [95% cluster CI] |
| --- | ---: | ---: | ---: |
| wrist x → 10-step object x | +21.15 mm | +13.04 mm | +17.23 [12.30, 22.06] mm |
| wrist y → 10-step object y | +8.51 mm | +17.99 mm | +13.05 [1.66, 22.05] mm |
| wrist z → 10-step object z | +2.41 mm | +12.80 mm | +7.40 [0.10, 14.66] mm |
| wrist x → 10-step contact | +5.46 pp | +4.55 pp | +5.02 [1.99, 8.55] pp |
| wrist z → 10-step contact | −8.66 pp | −2.38 pp | −5.64 [−10.29, −1.60] pp |

The x-axis result meets the pre-registered ≥5 mm same-sign/two-seed
and pooled-CI gate. Randomization supports a population intervention
effect on reached states, not an individual same-state counterfactual.
This does not show that any Cm can predict heterogeneous effects or
improve grasp policy.

## Decision update

Train a new action-conditioned 10-step Cm on the two run datasets and
test its action-effect predictions on a fresh seed with a state-only
capacity control. Only if held-out action information is usable should
this route be connected to policy decisions.
