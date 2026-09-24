# P-20260924-cm-crossaxis-primer

Date: 2026-09-24. Branch: `agent/cm-crossaxis-primer`.
Classification: Decision.

## Question

Can a lateral pre-adjustment improve the contact-supported lift from
a subsequent wrist-z action? The two-step x effect is learnable but
not task-aligned by itself; repeated z had uncertain additional
effect. This is the cheapest test of a task-aligned mixed-axis sequence
before a new Cm/planner build.

## Fixed randomized design

Use the pinned self-trained e260 actor, reconstructed motion,
new simulator seed173, 64 environments and 16 contact intervention
times 50..200 (stride10). Eligible states have unclipped candidate
x/z actions. Randomize equally among three groups:

* code −1: wrist-x −0.1 at t, then wrist-z +0.1 at t+1;
* code +1: wrist-x +0.1 at t, then wrist-z +0.1 at t+1;
* code +2 (control): actor action at t, then wrist-z +0.1 at t+1.

The actor controls all other actions. Record actual first/second
executed doses and ten-step object/contact outcomes. This is a
population randomized effect, not an individual counterfactual.

Primary metric: step-stratified mean difference (x− primer minus
control) in signed ten-step object z displacement multiplied by the
fraction of the ten follow-up steps with hand-object contact,
reported in mm. Environment-cluster bootstrap 500 draws.
Continue to a mixed-axis sequence Cm only if point gain ≥5mm,
95% CI lower bound >0, and x− contact fraction is no more than 5pp
below control. If point ≥5mm but CI crosses zero, repeat once on a
new seed and pool with prespecified step/env clustering; otherwise
stop. x+ primer is secondary and must not be selected post hoc on
seed173 if x− fails. Passing is not Cm policy utility.

One GPU, ≤30 min, ≤100 MB; stop on input drift, dose clipping,
incomplete cells, nonfinite tensors, GPU conflict or budget overrun.
Actor SHA `16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`;
motion manifest SHA `2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038`.

## Result

Pending.
