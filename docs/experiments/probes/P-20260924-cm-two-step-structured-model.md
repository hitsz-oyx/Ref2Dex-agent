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

Pending.
