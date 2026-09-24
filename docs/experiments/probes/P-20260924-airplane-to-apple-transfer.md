# P-20260924-airplane-to-apple-transfer

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe, not Validation.

## Question

Does the current airplane-trained, self-trained actor have enough zero-shot
ability on a different object to serve as the initial cross-object Cm policy?
If it fails, is that more likely an actor transfer gap or an invalid apple
simulation/reference task?

## Minimal comparison and decision

Use one reconstructed `s7_apple_lift` input, 64 first-complete episodes per
arm, fixed actor checkpoint `agent_v139_s3_standard_s70_e260` e260, early
termination disabled. Arm 1 executes the actor (seed174); arm 2 executes the
reference-action controller with lead 1 (seed175), retaining the same actor
checkpoint only to initialize the evaluator. The reference arm is a data/task
feasibility diagnostic, not a matched policy-utility control. The metric is
held-lift success: object rises >=3 cm with hand+object contact for >=5
consecutive steps.

If reference action is strongly positive but actor is <20%, prioritize
multi-object actor training before testing Cm policy utility. If both are
<20%, audit collision, retargeting and object-specific reference feasibility
before treating poor transfer as a learning result. If actor is >=20%, it may
support an early Cm-on/off cross-object Probe, but no claim without matched
control and multiple held-out object identities.

Budget: one idle GPU, <=20 min per arm, <=1 GB output. Stop on incomplete
episodes, mesh/provenance mismatch, GPU conflict, or non-finite results.

## Observed so far

Actor arm completed: apple 3/64 = 4.69% held-lift, mean contact fraction
0.68381, mean maximum contact lift 8.9 mm. This is a single-object,
single-seed Probe and does not by itself distinguish actor transfer from
reference/physics invalidity. Reference arm pending.
