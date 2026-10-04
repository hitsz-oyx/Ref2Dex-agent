# Decision: persistent hidden density feasibility

Question: does varying a persistent physical property create task decision
headroom worth a history-conditioned, action-conditioned Cm experiment?
This retains the airplane and own-policy training utility mission.

Evidence: actual8tick privileged successor has a31.07% Brier advantage over
directQ, but tested learned-successor policy recipes fail matched utility gates.
Current state beats motion/time-only prediction. The source airplane is
2.593613g at density20kg/m³. The current actor's force channels divide by SDK
actual mass: hiding mass requires pinning that denominator to nominal mass.
The one-tick force-disturbance family is closed; this is a persistent density
change from creation, not a force dose/window scan.

Choice: one fixed two-level density contrast20 versus1000kg/m³ (factor50;
2.594 versus129.681g), all geometry/COM unchanged, mass AND inertia scaled
before prepare_sim. The heavy level is chosen from density, before new task
outcomes, not optimized to obtain a success drop. First verify GPU free-body
force/torque response. Then one768-environment seed701 panel, balanced32 per
motion×policy×load. Primary arm is the already-fixed own coldQ checkpoint used
in the corrected-physics sensitivity run, with P0/Cm/action-off diagnostic arms.
No actor/Cm updates. Complete105tick full-mesh holding criterion remains.

Cost: one idle GPU, <=600s and512MiB; no official actor actions or external
project changes. Engineering failure blocks native probe. If nominal coldQ
>=25%, heavy coldQ is10..80%, and pooled nominal-minus-heavy >=15pp, the Probe
is PROMISING and a history-versus-current physical-prediction learnability
screen becomes worthwhile. Otherwise close this exact density contrast without
dose, seed, force-axis, or checkpoint rescue. One seed cannot validate robustness
or policy learning benefit. No new outside authorization is needed.

Prior-art boundary: RMA (RSS2021) uses a base policy plus online adaptation;
Qi et al.'s in-hand rotation work (CoRL2022/PMLR2023) uses proprioceptive history
to adapt across object properties. Generic latent-parameter/history adaptation
is not new. Only primary abstracts have been checked here, not full-method
equivalence or an exhaustive novelty review:
[RMA](https://www.roboticsproceedings.org/rss17/p011.html),
[In-Hand Object Rotation via Rapid Motor Adaptation](https://proceedings.mlr.press/v205/qi23a.html).
