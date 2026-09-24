# P-20260924-s1-task-aligned-option

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe — Option A physical gate.

## Question

The same sustained grip+lift option failed under the multi-object e260 actor,
including a +1.45 mm airplane subgroup. Does it nevertheless create a useful
H20 load-bearing distinction under the single-object `s1_airplane_lift` e140
self-trained actor that contributes to the strong V1.28 system and has room
for improvement as one network (about 52% on evaluation seed76)?

## Fixed protocol and decision

At pre-contact states, randomize ten policy steps of:

* grip+lift: additive +0.1 wrist-z and +0.2 on each independent finger-flexion
  command at every step;
* lift-only: the same +0.1 wrist-z and no added finger flexion.

Use the pinned e140 checkpoint (SHA
`5d1f50a21409df5a09f0a471d115d2f06eedce6bc53855a81acf3cc268e6e1a7`),
only `s1_airplane_lift`, 128 environments, seed191, intervention steps 50..190
stride20, and H20 followup. Commands remain bounded to [-1,1]. The primary
outcome is survival-adjusted H20 object-z displacement weighted by contact
fraction.

Replicate on seed192 before training any Cm only if both arms have >=100
treated rows, grip+lift minus lift-only is >=5 mm, the environment-cluster
95% interval has positive lower bound, and the contact-fraction contrast is
nonnegative. Otherwise stop this action family and do not train the proposed
task-aligned Cm. This Probe tests a physical opportunity, not Cm utility.

One idle GPU, <=30 minutes, <=100 MB. Stop on checkpoint/source/dose drift,
GPU conflict, non-finite state, incomplete followup or wall-budget overrun.

## Result

`agent_s1_task_aligned_option_s191_h20_n128` completed with 572 treated
contact states: 287 grip+lift and 285 lift-only. Grip+lift minus lift-only
contact-supported H20 object-z displacement was **-8.20 mm**, with
environment-cluster 95% interval **[-14.08, -2.49] mm**. The contact-fraction
contrast was -2.05 percentage points. No intervention reset was observed.

The >=5 mm, positive-CI and nonnegative-contact gate failed, with the primary
effect significantly in the wrong direction. Classification: `UNPROMISING`.
Do not run seed192 and do not train a Cm for this action family. Combined with
the prior multi-object result (+0.15 mm pooled, +1.45 mm airplane subgroup),
this closes the sustained additive finger-grip/wrist-lift option route under
the tested self-trained actors.

The strict `effect_report_v2.json` audit also verified ten treated steps,
cumulative +1.0 wrist-z increment in both arms and cumulative +2.0 increment
for every selected finger in the grip arm.

A later read-only recomputation from the pinned 616 KB `transitions.pt`
reproduced the 287/285 arm counts, −8.1980 mm primary contrast and
environment-cluster interval exactly. Both arms had zero resets. The secondary
unweighted H20 object-z contrast was also negative (−14.49 mm), so the sign
of the primary result is not created solely by contact-fraction weighting.
This remains a single-seed physical Probe, not a Cm policy comparison.
