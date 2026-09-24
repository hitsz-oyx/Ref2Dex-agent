# P-20260924-train8-h10-action-value

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe / candidate-action availability.

## Question and fixed gate

The H5 raw Cm predicts held-out *mean* effect but cannot rank states.
Before designing a longer-horizon Cm, test whether the simple ±0.1
wrist-z candidate family even creates distinct **H10 contact-supported
lift** outcomes across the eight training object identities. Collect
official-origin randomized interventions on train8 seed 188, steps
50..200 stride 10, H10; the official actor is only a data generator.
For each identity (group duplicated cubesmall motion IDs), estimate
plus-minus mean object dz weighted by followup hand/object contact
fraction, with environment-cluster bootstrap 95% intervals.

If at least three identities have absolute effect >=10 mm with CI
excluding zero **and** those reliable effects include both positive
and negative directions, an object-conditioned H10 selector has a
physical opportunity and warrants a new Cm target. Otherwise abandon
this z-only option family rather than fit another network to it.
One idle GPU <=15 minutes, <=100 MB output; CPU analysis <=5 minutes.
This Probe cannot establish final policy utility.

## Result

Status: `UNPROMISING` for the object-conditioned wrist-z H10 selector.
The hash-pinned seed-188 source contained 1,024 rows and 984 assigned
interventions. The legacy single-axis collector clipped one negative dose
from -0.1 to -0.0685 because the base action was already -0.9315; all other
assigned doses were exact and no other action axis changed. The analysis
therefore kept the randomized assignment as an intention-to-treat contrast
and reports this 983/984 full-dose compliance rather than dropping the row.

Only cup (+21.44 mm, environment-cluster 95% CI [+3.29,+44.26]) and
waterbottle (+20.10 mm, [+4.89,+44.83]) met the fixed absolute 10 mm and
nonzero-CI criterion. The other six identities were not reliable by the
fixed rule. Thus only two identities passed, and both signs were positive;
the required at least three identities with both positive and negative
directions failed. Do not fit another object-conditioned z-only selector or
attach this option family to PPO. This does not reject longer-horizon Cm in
general; it removes the physical-opportunity premise for this particular
candidate set. Report:
`outputs/CmResidual/agent_train8_h10_action_value_s188/report.json`.
