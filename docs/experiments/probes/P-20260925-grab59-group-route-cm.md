# P-20260925-grab59-group-route-cm

- Classification: Decision Probe.
- Substrate: the fixed simulator-object-ID route over 59 lift-like motions,
  29 objects, and ten self-trained experts. This is the current usable
  self-trained grasp substrate after the uniform 660-motion actor failed.
- Stable route config: `src/task/CmResidual/configs/grab59_group_route.json`.

## Question and matched arms

Does Cm improve the specialist route when it ranks either the frozen expert
actions or local residual proposals? The Cm-off arm executes the same fixed
route and source checkpoint. Every arm uses one first episode per 64 parallel
environments, disabled early termination, and the strict held-lift definition:
object rise of at least 3 cm while hand-object contact persists for five
consecutive steps.

The first test used the existing V1.37 Cm checkpoint (`396f5e...ed204a`) to
rank all expert actions on seeds 244 and 245. The second test collected 67,200
route-specific one-step transitions from seed 244, trained a relative-wrist
CmLite checkpoint (`64759650...4995c`), and evaluated local proposals behind a
five-step stable-contact gate on new seeds 245--247. Before seed 247, the
route-specific replication gate was set to at least +4/192 held-lifts and no
pooled contact regression.

## Results

The fixed route itself produced 20/256 held-lifts on the first four baseline
runs (seeds 244--247), with mean hand-object contact of 25.5%.

| Cm arm | Seeds | Held-lift | Mean contact | Interpretation |
| --- | --- | ---: | ---: | --- |
| off | 244--245 | 11/128 | 25.4% | matched control for old Cm |
| V1.37, all experts, instant gate | 244--245 | 10/128 | 24.6% | no gain |
| off | 245--247 | 13/192 | 25.7% | matched control for route-specific Cm |
| route-specific Cm, local + stable | 245--247 | 16/192 | 27.3% | +3/192, below the +4 gate |

The route-specific model replaced the base action on 6,499, 5,555, and 5,236
of roughly 68,800 scored steps in seeds 245, 246, and 247. The implementation
was therefore active, but the held-lift increase was small and did not pass
the predeclared replication gate. The result is `UNCLEAR` for a distribution
adaptation effect and is not evidence of Cm policy utility.

## Decision and boundary

Do not claim that either V1.37 action ranking or route-specific local residual
Cm improves GRAB grasping. Keep the group route as the reproducible baseline
substrate and stop tuning this one-step goal-distance selector. The next Cm
Probe should change the credit target toward contact-supported lift or a
longer-horizon outcome, while preserving the same fixed route and matched
Cm-off arm. None of these results is a full 660-motion or raw-GRAB success
rate.

## Artifacts

- Baseline results: `outputs/CmResidual/agent_grab59_group_route_s{244,245_r2,246,247}/results.json`.
- Old-checkpoint Cm results: `outputs/CmResidual/agent_grab59_group_route_cm_experts_s{244,245}/results.json`.
- Route-specific Cm results: `outputs/CmResidual/agent_grab59_group_route_cm_route59_local_s{245_r2,246,247}/results.json`.
- Route-specific training data and checkpoint:
  `outputs/CmResidual/agent_grab59_group_route_transitions_s244/transitions.pt`
  and `outputs/CmLite/V1.route59_s244_e40_r2/best.pt`.
