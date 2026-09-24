# P-20260924-cm-two-step-structured-model

Date: 2026-09-24. Branch: `agent/cm-two-step-sequence`.
Classification: Decision.

## Why this one architectural revision

Two-step physical effects replicated on new seeds 168/169, but the
first raw-concatenation Cm underpredicted the held-out second-step
interaction (6.28 mm vs 19.84 mm; 0.316×), even though factual x RMSE
improved 11.4% over state-only. This suggests ordinary factual
regression can suppress the small conditional second-action signal.
Do not tune that model on seen seed169.

## Fixed Probe

Use both randomized seeds168/169 for training (four signed duration
arms), and a genuinely unseen simulator seed170 for testing. Replace
the raw-concatenation Cm with a single structured-effect model:
shared pre-treatment state/base-action trunk, plus separate learned
first-dose and second-dose translation/contact effects multiplied by
the planned signed ±0.1 dose. Same-size state-only control zeroes both
doses; initialization, optimizer, batch schedule and 500 updates are
matched. The model never consumes the post-treatment second-step base
actor action. No contrast loss, architecture search or test-seed tuning.

Pre-registered gate: on seed170, observed two-vs-one signed x
interaction must retain positive sign; the action-conditioned model
must improve factual ten-step world-x RMSE by ≥5% over state-only,
and predicted interaction must have the same sign and magnitude
0.5–1.5× the randomized observed value. Otherwise stop this model
route, rather than proceed to online planning. Passing is still only
model `PROMISING`, not policy utility.

One GPU for seed170 collection (<30 min, ≤100 MB), CPU 2 threads for
training (≤10 min, ≤100 MB); stop on input drift, dose clipping,
leakage, nonfinite output or budget overrun.

## Result

Status: `PROMISING` for held-out two-step wrist-x effect prediction,
not policy utility. Code commit `42b1345`. The seed170 physical
interaction retained positive sign: +18.96 mm, 95% environment-cluster
CI [+9.06,+27.65] mm. `agent_two_step_structured_cm_s168169_train_s170_test`
completed 500 CPU updates on 1485 seed168/169 train rows and tested
only on 170. Factual ten-step world-x RMSE improved 11.84% versus
same-size state-only. Predicted second-step interaction was 0.843×
the randomized observed value; both pre-registered model gates passed.
The report and both checkpoint SHA256s are in the run directory.

Exploratory task-alignment audit on the already-seen seed170 found
the wrist-x plus-minus contrast in ten-step object z was negative
(one step −4.71 mm, two steps −9.52 mm); the structured model also
predicted a negative contrast. Wrist-x response is real and modeled,
but pushing x alone is not a justified grasp-lift planner objective.
Do not run an x-only online selector merely because the x-model gate
passed. Next cheapest decision is a new randomized two-step wrist-z
physical Probe, with new seed(s) and explicit contact tradeoff.
