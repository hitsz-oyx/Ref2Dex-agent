# P-20260924-sustained-grip-lift-option

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe.

## Question

A one-step finger primer increases contact but does not improve ten-step
lifting, while one-step wrist displacement loses its apparent benefit under
closed-loop followup. Is a *sustained, task-aligned action option* a
nontrivial choice that Cm could eventually evaluate before execution?

## Minimal protocol and gate

Under pinned self-trained e260 actor on airplane/mug/toothpaste only, at
pre-contact states randomize 10-step additive options:

* arm `+1`: add +0.1 wrist-z and +0.2 to each of five independent finger
  flexion commands at each policy step;
* arm `-1`: add the same +0.1 wrist-z but no extra finger flexion.

Each resulting policy command is bounded to [-1,1]; report actual executed
increment, including saturation. Follow the ten option steps plus ten
actor-only steps, total H20, without early episode termination. Use 128
environments, intervention steps 50..190 stride20, seed190. Same train
object split; do not use apple to choose this route. One idle GPU, <=30
minutes, <=100 MB.

Primary outcome: survival-adjusted H20 object-z displacement weighted by
hand/object contact fraction. Continue only if the grip+lift minus lift-only
pooled point effect >=10 mm, at least two objects positive, >=50 treated
rows/object, and pooled contact-fraction effect nonnegative. If passed,
replicate before training Cm and compare Cm-gated option against both base
actor and unconditional option. If failed, stop small action-option
engineering and move to policy representation/credit assignment rather than
claiming Cm can choose among these actions.

This is a Probe; neither arm alone proves Cm utility.

## Result

Status: `UNPROMISING` for this bounded ten-step option family. An 8-env
engineering smoke completed, then the 128-env train-object run
`agent_crossobject_sustained_option_train_s190_h20_n128` completed with
700 treated states: airplane 231, mug 260, toothpaste 209. Both arms
executed a full cumulative +1.0 wrist-z command increment over ten steps;
the grip arm also executed full cumulative +2.0 per selected finger (no
finger-command saturation in treated records).

Grip+lift minus lift-only contact-supported H20 object-z contrasts were
+1.45 mm airplane, +0.04 mm mug, -1.14 mm toothpaste, pooled +0.15 mm
with environment-cluster 95% interval [-1.58,+1.89] mm. The gate of
>=10 mm and two positive objects failed. Absolute mean H20 object-z
displacement was near zero or negative under both arms, despite the
command dose. Additional grip increased pooled H20 contact fraction by
about +3.39 pp, but did not lift the object. Do not replicate, train Cm
on this option, or call it a policy improvement. Together with the H1/H10
diagnostic, this favors a higher-level policy/credit-assignment probe over
further small wrist/finger action engineering.
