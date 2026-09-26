# P-20260924-object-specialist-router

- Classification: Decision Probe.
- Cm: off. Every expert checkpoint is self-trained; no official actor.
- Route: fixed object identity from simulator metadata, hence a privileged
  diagnostic baseline, not an observation-driven deployable policy.

## Question and decision

Can a fixed portfolio of already trained self-trained actors demonstrate
simultaneous successful grasp on several corrected GRAB trajectories where
one shared actor scored 16/128? Freeze the route using prior seeds201/202
and duck seeds206–208: airplane e260, duck specialist e340, mug train5
e320, toothpaste balanced e360, alarmclock mixed12 e300; other objects use
e260. Evaluate first full episodes on entirely new seeds211–213, 64
environments each, with exactly the same 12 motions and disabled early
termination. Compare pooled held-lift with the previous best fixed actor
(21/128 on seeds201/202) only as exploratory context because the seeds
differ. Report every object's denominator, contact and failure count; do
not infer broad GRAB coverage from a pooled win.

Pass the *coverage* gate only if at least four object identities each
achieve >=25% held-lift across seeds211–213 and pooled success >=25%.
If passed, specialists are a viable multi-trajectory substrate; move to
observation-driven routing and then matched Cm value tests. If not, add or
repair object specialists before a Cm router. One or two idle GPUs,
<=30 minutes, <200 MB output. Stop on checkpoint/config/data drift,
GPU conflict, inconsistent route or incomplete evaluation.

## Results

The checkpoint loader initially rejected compiled training key prefixes,
and the first retry incorrectly indexed the ten object names by twelve
motion IDs. Both were implementation failures; neither produced a valid
scientific result. Commits `1b5426d` and `99c4491` repaired these issues.
The final frozen route was evaluated on fresh seeds211–213 with one GPU,
64 first full episodes per seed and disabled early termination. All three
run manifests report `COMPLETED`. Their held-lift counts were 17/64,
17/64 and 21/64, totaling **55/192 (28.65%)**. Mean hand-object contact
fractions were 0.306, 0.331 and 0.371.

| Object | Held-lift / episodes | Mean contact fraction |
| --- | ---: | ---: |
| airplane | 17/45 | 0.362 |
| alarmclock | 0/18 | 0.179 |
| apple | 2/15 | 0.254 |
| cubesmall | 1/18 | 0.200 |
| cup | 0/18 | 0.066 |
| duck | 12/18 | 0.674 |
| mug | 9/15 | 0.507 |
| phone | 0/15 | 0.124 |
| toothpaste | 14/15 | 0.775 |
| waterbottle | 0/15 | 0.211 |

Four identities (airplane, duck, mug, toothpaste) exceeded 25% and pooled
success exceeded 25%, so the prespecified coverage gate **passed**.
Interpretation: `PROMISING` for a specialist portfolio as a multi-trajectory
substrate. It is a single fixed route selected from previous Probe seeds,
not a matched comparison against a single actor on the same new seeds,
nor evidence for full GRAB coverage or Cm benefit. The previously weak
alarmclock route produced 0/18; the portfolio still has six identities
below 25%. Next test an observation-driven object/expert route and compare
Cm against a matched object-only router before claiming Cm policy utility.

Run artifacts: `outputs/CmResidual/agent_multitrajectory_object_router_s211_fix3/`,
`..._s212_fix3/` and `..._s213_fix3/`, with corresponding
`outputs/CmResidual/router_s21{1,2,3}_fix3.log` mapping each motion ID to
its object and expert. These are local converted trajectories, not the full
GRAB dataset.
