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

Before viewing seed172, define the borderline repeat rule precisely:
pool seed171 and new seed172 with each seed's intervention step as a
separate stratum, resample environments independently within each
seed, and apply the same ≥5mm/CI>0 and contact ≥−5pp gate to the
pooled estimate. If it fails, stop; do not add a third seed to search
for a passing subset.

One GPU, ≤30 min, ≤100MB; stop on source drift, clipping, incomplete
cells, nonfinite state, GPU conflict or budget overrun. Actor SHA
`16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`;
motion manifest SHA
`2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038`.

## Result

Status: `UNCLEAR` for an additional wrist-z physical interaction;
pre-registered pooled gate failed, so stop this local sequence route.
Collector code commit `25ad3bc`, pooled decision code commit `77ffbe9`.
The 16-env z smoke completed and passed exact first/second dose audit.
Both 64-env randomized data runs `agent_two_step_z_s{171,172}_n64`
completed on physical GPU6; each has 16 contact intervention times.
Single-seed analysis run IDs `agent_two_step_z_s{171,172}_analysis`
and fixed pooled run ID `agent_two_step_z_s171172_pool` completed.
Commands, source SHA, status and results are in each output manifest.

| Seed | Second-step z interaction | Environment-cluster 95% CI | Two-step minus one-step contact |
| --- | --- | --- | --- |
| 171 | +7.00 mm | [−2.59,+16.32] mm | −3.47pp |
| 172 | +5.65 mm | [−3.87,+15.40] mm | −1.76pp |
| pooled | +6.33 mm | [−0.93,+12.86] mm | −2.61pp |

The point interaction exceeds 5mm and contact cost remains within
the −5pp gate, but the pooled 95% interval includes zero. No third
seed or dose tuning is allowed by this decision rule. This does not
refute all temporal Cm representations; it only withholds permission
to train a wrist-z repeated-dose sequence Cm or run z planning under
this fixed design.
