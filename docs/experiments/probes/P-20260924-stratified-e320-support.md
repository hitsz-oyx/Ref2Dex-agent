# P-20260924-stratified-e320-support

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Blocker/Decision Probe — representation data support.

## Question and fixed decision

The mixed train5 replay failed the 4/5 support gate partly because each
non-cubesmall identity received only 10–11 environments. A 64-environment
waterbottle-only replay of the same frozen e320 actor exposed 138 positive
H10 windows across 14 environments, while subsequent PPO destroyed them.
Does equal per-object collection supply adequate frozen-e320 representation
data without additional policy training?

Reuse the existing mixed train5 e320 replay for airplane, cubesmall and mug;
reuse the fixed waterbottle-only e320 replay; collect one matching
toothpaste-only e320 replay on seed178. Apply the identical positive/failure
definitions and support thresholds. Each object is evaluated from exactly
one pinned source; no rows from e340/e360 enter the pool.

If all five identities pass, proceed to a dense interaction representation
Probe using object-balanced sampling plus raw, action-blind and
action-shuffled controls. If any identity fails, stop and redesign collection
or target before fitting. This only establishes label support, not Cm
accuracy, generalization or policy utility.

One idle GPU <=10 minutes for the single missing replay; CPU <=10 minutes;
<100 MB. Stop on source/split/checkpoint drift or incomplete episodes.

## Result

`outputs/CmResidual/agent_stratified_e320_support_s178/report.json` passed
the fixed gate for all five identities. Positive/failure window counts were:

* airplane: 979 / 1679, positive states in 7 environments;
* cubesmall: 252 / 1932, positive states in 10 environments;
* mug: 2974 / 3146, positive states in 10 environments;
* toothpaste: 914 / 12040, positive states in 15 environments;
* waterbottle: 138 / 3382, positive states in 14 environments.

Classification: `PROMISING` for data support only. Proceed to the frozen dense
interaction representation Probe. These overlapping H10 windows are not
independent evidence of Cm accuracy or policy utility.
