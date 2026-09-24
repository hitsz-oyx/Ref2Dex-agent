# P-20260924-cm-two-step-physical-effect

Date: 2026-09-24. Branch: `agent/cm-two-step-sequence`. Classification: Decision.

## Decision question

After the H10 multi-axis train-time auxiliary Probe failed its matched
policy gate, does a second controlled wrist-x action add physically
meaningful information beyond a single perturbation? If not, do not
invest in a sequence-conditioned Cm and short-horizon planner.

## Minimal randomized Probe

Use the fixed self-trained e260 actor, reconstructed motion, one new
simulator seed 168, 64 environments and 16 intervention times
50..200 (stride 10). At actual current contact and with an unclipped
wrist-x action, independently randomize four equally allocated cells:
`+1/-1` applies wrist-x ±0.1 at the first step only; `+2/-2`
applies the same ±0.1 at the first and next step. Base actor controls
all other steps. Follow each intervention to ten steps. Record the
actual first/second action, object state and contact at every endpoint.
This is a randomized population sequence effect, not a same-state
counterfactual; second-step actor base action is post-first-treatment.

Primary signal is the interaction in ten-step object x displacement:
`(two+ − two−) − (one+ − one−)`, step-stratified and resampled by
environment. Gate to a sequence-Cm model only if point interaction
≥5 mm, environment-cluster 95% CI lower bound >0, and two-step
contact fraction is not >5pp below one-step averaged over signs.
If borderline, collect one additional new seed before deciding;
if clearly below gate, stop. No policy utility claim from this Probe.

One GPU, one 64-env rollout, wall ≤30 min, output ≤100 MB;
stop on input drift, dose clipping, nonfinite state, incomplete cells,
GPU conflict or budget overrun. Fixed source actor SHA
`16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`
and motion manifest SHA
`2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038`.

## Result

Pending.
