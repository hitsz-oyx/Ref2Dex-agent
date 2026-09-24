# P-20260924-cm-two-step-z-effect

Date: 2026-09-24. Branch: `agent/cm-two-step-sequence`.
Classification: Decision.

## Why this Probe

Two-step wrist-x action has reproducible x effect and a held-out
structured Cm can predict it, but x-positive actions also tend to
reduce ten-step object z on the seen development seed170. Lateral
effect alone is not a justified held-lift planner target. Test whether
the same sequence idea is physically useful in task-aligned wrist z
before training another model or launching online policy runs.

## Fixed design and gate

Use the same self-trained e260 actor and 64-env randomized four-arm
design on new seed171: wrist-z ±0.1 for exactly one or two consecutive
steps, assigned at 16 contact intervention times (50..200, stride10).
Record exact executed doses; follow each event ten steps. Primary
interaction is `(two+−two−)−(one+−one−)` in ten-step object z
displacement, step-stratified with 500 environment-cluster resamples.
Task-alignment gate: interaction ≥5mm with 95% CI lower bound >0,
and average two-step contact fraction no more than 5pp below one-step.
If borderline, repeat on one new seed; if negative or contact cost
exceeds gate, stop wrist-z sequence modeling and do not tune dose on
seed171. Passing authorizes only a held-out z-sequence Cm model Probe,
not an online policy claim.

One GPU, ≤30 min, ≤100MB; stop on source drift, clipping, incomplete
cells, nonfinite state, GPU conflict or budget overrun. Actor SHA
`16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`;
motion manifest SHA
`2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038`.

## Result

Pending.
