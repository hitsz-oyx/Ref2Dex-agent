# P-20260924-train5-control

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe / Cm-off distribution and transfer control.

## Question and cheapest useful test

Does expanding corrected, physically feasible training motions from
three to five distinct object identities make the self-trained actor
less object-specific on untouched alarmclock, and provide a meaningful
multi-object transition distribution for the next Cm architecture test?
Use the exact e260 airplane checkpoint, matched prior train3 PPO reward,
curriculum, seed 179 and 60 additional epochs to e320, replacing only
the training motion root with the pinned train5 split. The initial
alarmclock e260 control is 0/64 held-lift on seed 174. Evaluate the new
e320 actor on train5 seed 178 and alarmclock seed 174, all 64 first
episodes, no Cm. Save full transitions only if the train5 actor exhibits
some nontrivial contact-supported lift; otherwise avoid training a Cm
on an uninformative policy distribution.

If alarmclock held-lift reaches >=8/64, treat expanded data as a
`PROMISING` transfer direction and retain this actor as the matched
Cm-off control. If it remains <8/64 but train5 contains object-specific
successes, use those train transitions to diagnose action-conditioned
load-bearing dynamics; do not claim cross-object improvement. If the
new objects fail physically, revisit reward/data before Cm PPO.
One idle GPU, <=20 min, <2 GB checkpoints/logs; abort on data hash,
GPU conflict, non-finite training or incompatible resume checkpoint.
This is not a Cm effect test and no formal claim follows from one seed.

## Result

The train5 Cm-off run completed to e320. On train5 seed 178, held-lift
was 19/64, mean maximum contact-supported lift 205 mm. DExplore's
built-in hard-object sampler duplicates the `cubesmall` motion, yielding
six motion IDs for five object identities: cubesmall 0/21 (IDs 0 and 5),
mug 9/11, toothpaste 4/11, waterbottle 0/11, airplane 6/10.
The untouched alarmclock held-out evaluation seed 174 remained **0/64**,
versus e260 0/64, although mean maximum contact-supported lift rose
from 6.06 to 13.07 mm. The preregistered >=8/64 transfer gate failed;
mark this 60-epoch data-only route `UNPROMISING` for held-out transfer.
The positive train-object episodes permit transition collection, but
their hard-object duplication must be accounted for in any object-level
analysis. No Cm benefit or negative Cm conclusion follows.
