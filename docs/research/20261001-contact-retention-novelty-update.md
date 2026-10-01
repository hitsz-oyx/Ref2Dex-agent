# Contact/retention idea screen during baseline training

Read primary-source abstracts on2026-10-01. This is a targeted novelty screen,
not a systematic literature review or independent reproduction of author claims.
It does not change the running baseline recipe or establish our method's novelty.

| Primary source | What it already covers | Constraint on our next idea |
| --- | --- | --- |
| [DexTacWAM, Yuan et al., v1,21Sep2026](https://arxiv.org/abs/2609.24976v1) | Finger/pose-aware tactile encoding and jointly predicted visual/contact dynamics; matched tactile features without contact forecasting are an ablation | Simply forecasting contact evolution rather than conditioning on it is not a new claim |
| [WHIRL, Yin et al., v1,5Sep2026](https://arxiv.org/abs/2609.06009v1) | Latent dynamics/reward/termination plus future human-intervention probability, with actor-side predictive risk shaping | Merely adding a failure/intervention risk head to a world model or actor loss is insufficient novelty |
| [DexTouch-WM, Qin et al., v2,18Sep2026](https://arxiv.org/abs/2609.20649v2) | Compatible human/robot action/touch layouts, joint RGB/tactile dynamics, surrogate policy evaluation and synthetic policy-learning trajectories | Generic human-to-robot contact pretraining or model-generated policy data is occupied territory |

Our inference: a future retention-aware Cm design needs a distinct, falsifiable
mechanism and matched real policy-learning improvement. A contact/survival head
alone would be a baseline, even if it succeeds on the current one-object task.
The completed experiments motivate testing whether history/attributed contact
and task-aligned multi-step consequences help, but they have not established
why earlier predictors failed. No model-utility experiment is launched before
the frozen actor baseline feasibility gate closes.

Measurement issue to carry forward: the current two force proxies do not
attribute hand-object versus hand-table/object-table forces. Object-root rise
plus both proxies can also include supported tilting of an elongated object.
This limits what the75-step proxy endpoint can establish. Any future formal
grasp claim should audit object-table clearance, hand-object geometry/contact
attribution and physical trajectories, with an independently frozen criterion.
Do not retroactively change a failed gate or assume that every counted hold is
a force-closure grasp. Existing compact traces omit object orientation, so
that distinction is not identifiable from height/proxy labels alone.
