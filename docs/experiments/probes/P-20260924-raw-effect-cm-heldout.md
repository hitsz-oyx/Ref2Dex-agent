# P-20260924-raw-effect-cm-heldout

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe / redesigned Cm information test.

## Question

An action-conditioned raw state Cm causal head trained on seven of
eight object identities predicted the omitted object's **mean** H5
randomized wrist-z effect with 7.96 mm MAE, versus 15.28 mm for a
constant-effect control and 14.09 mm for the six-region geometry head.
Does a model fitted on all eight train identities retain usable
action-before-outcome information on untouched alarmclock, and can
its score rank state-dependent treatment effects there?

## Fixed design and decision

Collect one seed-187 ±0.1 wrist-z, H5 randomized intervention from
the read-only official actor on alarmclock, steps 50..200 stride 10,
64 environments; official weights remain a data generator, never the
final policy. Keep alarmclock labels sealed until fitting a 300-step
raw causal head on the pinned train8 seed-186 source. The model input
is only pre-action state/action; the signed effect head predicts the
plus-minus outcome difference. Compare mean predicted and randomized
actual treatment effect, and evaluate pre-score top versus bottom
quartile randomized uplift with environment-cluster bootstrap.

Advance this raw-action Cm to a *short-horizon policy-candidate* check
only if: (1) held-out mean effect sign matches, (2) absolute ATE error
<=10 mm, and (3) top-minus-bottom quartile realized effect >=10 mm
with 95% bootstrap lower bound >0. Otherwise classify `UNPROMISING`
or `UNCLEAR` and do not connect it to PPO. Even a pass is a Probe,
not long-horizon grasp or Cm policy utility. One GPU <=15 minutes,
<=100 MB output; CPU fit <=10 minutes.

## Result

The official-origin alarmclock seed-187 randomized source completed.
The eight-object raw causal head was fitted and frozen before opening
alarmclock labels. It predicted mean H5 wrist-z ± effect +26.29 mm;
randomized actual was +21.07 mm (error 5.22 mm, correct sign).
However its top versus bottom score quartiles had actual effect
19.81 versus 16.68 mm: difference +3.13 mm, environment-cluster
95% interval [-7.58, +15.42]. The prespecified >=10 mm positive-CI
ranking gate **failed**. This is useful object-level effect prediction,
but not reliable within-object action selection; mark the route
`UNPROMISING` for the current H5 candidate-ranking purpose, and do
not attach it to PPO. The result does not refute other Cm targets or
longer-horizon model-based uses. Artifacts:
`outputs/CmResidual/agent_expert_alarmclock_randomized_s187_h5/` and
`outputs/CmResidual/agent_raw_effect_cm_alarmclock_s187/`.
