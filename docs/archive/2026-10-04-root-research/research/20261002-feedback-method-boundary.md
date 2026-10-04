# Predictive versus reactive contact support: primary-source boundary

Primary full-text method sections inspected2October2026. This is reading, not
independent reproduction or endorsement of reported performance.

| Primary source | Existing mechanism | Consequence for this branch |
| --- | --- | --- |
| [Abd et al., Direction of Slip Detection, AIM2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC7009943/), II-F/III-C | Tactile slip-direction classifier drives grasp-synergy tightening for downward slip and opening for transfer. | A slip-triggered curl reflex is established and cannot be a new contribution. Our simulator root/dq state is different from their tactile sensing. |
| [TouchWorld v2](https://arxiv.org/html/2607.07287v2),2.3/3.4/4.1 | Recent tactile/proprioceptive histories plus nominal action lookahead produce residual action windows; feedback is refreshed within a slower nominal policy. Residual learning uses demonstrated corrections. | Separating predictive world-model goals and fast reactive contact control is already described. Our four mechanical arms are much narrower and have no learned tactile residual. |
| [ContactGuard v1](https://arxiv.org/html/2608.13438v1),3.1--3.5/4.1/5 | Action-conditioned future visual latents are scored by a frozen linear failure probe at a pre-contact trigger; thresholded scores abort a policy chunk. Post-abort completion/recovery is explicitly outside its demonstrated scope. | Generic action-conditioned risk monitoring is occupied. A recovery controller would ask a different utility question, but that difference alone does not establish novelty. |
| [Foresight v1](https://arxiv.org/html/2606.23085v1),4.2--4.4/8--9 | Predictive action-conditioned latents feed a causal temporal detector trained with final rollout labels; functional conformal bands calibrate alarms using successful calibration trajectories. | Temporal failure scoring and calibrated alarm timing are established; a new threshold is insufficient. Alarm accuracy still must not be substituted for actual recovery or policy training. |

Inference: another forecast-plus-slip-trigger system has no obvious methodological
priority from its ingredients. A useful future contribution would need to resolve
a narrower technical question, such as action-specific recoverability rather
than only loss prediction, and demonstrate actual benefit against strong direct
state feedback/value learning under matched interaction and total pretraining
accounting. That question is currently unresolved; it is not a novel theorem or
an empirical result. Current fixed feedback Probe is negative and stays negative.
