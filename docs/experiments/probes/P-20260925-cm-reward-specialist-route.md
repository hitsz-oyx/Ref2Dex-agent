# P-20260925-cm-reward-specialist-route

- Classification: Decision Probe.
- Substrate: the fixed simulator-object-ID route over 59 lift-like motions,
  29 objects, and ten self-trained experts.
- Question: does a frozen route-specific CmLite progress reward improve the
  specialist actors when it is used during matched PPO continuation training?

The four group specialists (gC--gF) were continued from e360 to e400. The
Cm arm used the route-specific relative-wrist checkpoint
`outputs/CmLite/V1.route59_s244_e40_r2/best.pt` (SHA256
`64759650bb6aa11afd193fe33a3b6a7f20f4fe6d8864bd009e97251b3d04995c`) with
coefficient 1, positive-only reward, and the same contact/lift curriculum as
the Cm-off arm. The off arm used the same source checkpoints and training
budget. Group gC was first evaluated in isolation on seeds 96 and 97:

| Group gC arm | Held-lift | Mean contact |
| --- | ---: | ---: |
| Cm-off | 33/128 | 44.8% |
| Cm reward | 45/128 | 56.8% |

This local result was positive, so the four groups were assembled into matched
e400 route configs and evaluated over all 59 motions on seeds 248 and 249.
The route is still privileged by simulator object ID; it is not a single
660-motion actor or a raw-GRAB result.

| Full 59-motion arm | Seed 248 | Seed 249 | Pooled | Mean contact |
| --- | ---: | ---: | ---: | ---: |
| Cm-off | 10/64 | 8/64 | **18/128** | 29.9% |
| Cm reward | 6/64 | 10/64 | **16/128** | 30.6% |

The full-route matched result is therefore `UNPROMISING` for this reward
接法: the local gC gain did not transfer after the other specialist groups
were included, and the pooled held-lift rate decreased by 1.6 percentage
points. Contact fraction increased slightly, but that did not produce more
held lifts. Stop expanding this exact reward configuration.

Artifacts:

- `outputs/CmResidual/grab59_group_route_e400_{off,cm}.json`
- `outputs/CmResidual/agent_grab59_route_e400_{off,cm}_s{248,249}/results.json`
- e400 specialist checkpoints under
  `outputs/Dexplore/agent_grab59_group_g{C,D,E,F}_{off,cm}_*/train/`

The next Cm probe changes the credit gate rather than tuning this coefficient:
use predicted contact probability to provide pre-contact credit, with a
matched gC continuation first. This is a new hypothesis and will only be
extended to the full route if the isolated probe improves held-lift.

## Predicted-contact gate follow-up

The isolated gC condition did improve on the rerun pair (45/128 versus
29/128 held-lifts for Cm-off), so gD--gF were continued with the same e360 to
e400 budget and `--use-predicted-contact`. The full route was then evaluated on
the original seeds:

| Full 59-motion arm | Seed 248 | Seed 249 | Pooled |
| --- | ---: | ---: | ---: |
| Cm-off | 10/64 | 8/64 | **18/128** |
| Predicted-contact Cm | 10/64 | 4/64 | **14/128** |

This follow-up is also `UNPROMISING` at the route level. The gate changes the
credit timing, but the gC single-group gain does not transfer to the complete
specialist mixture; the route-level held-lift rate is 3.1 percentage points
lower. Stop this contact-gate variant before any coefficient sweep.
