# TRAIN-only task determinacy: after-certainty is not dominant

Source46d40fb, all20TRAINpanels547–566/15360trajectories, no final568/569 or laterTEST. No physics, neural fitting/forward, optimizer or label rewrite. Actual105labels rebuilt and60saved first-minibatch MCtargets/normalizedadvantages verified (maxadvantagedifference0).

| Variant | After-certainty rows | After-certainty Gaussian score-energy proxy |
|---|---:|---:|
| cm | 44.869% | 6.280% |
| state_only | 45.389% | 5.981% |
| none | 44.148% | 6.144% |

Predeclared50%decision criterion fails ALLvariants. Outcome **AFTER_CERTAINTY_NOT_DOMINANT**: do not make after-certainty filtering/history repair the primary next explanation or claim invalid gradients. Correct actualterminal105reward was already used. This arithmetic proxy does NOT reconstruct actor-network/PPO/clipping/joint-Adam gradients or their variance. MCscore gradients with action-independent baselines are not automatically biased by a history-dependent terminal target.

The strict goal still has a known absorbing failure and fixed105deadline. Only3currentlygeometrically-valid states exist after earlierfailure across all learned arms/20panels (0/1/2); this does not prove exact Markov adequacy of the70feature vector. Future value interfaces must carry taskclock/knownhistory, but evidence does not justify a newhistory-only learning campaign.

Completed34.258s, tiny statistic artifact, all pinned source/data/checkpoint hashes unchanged. Next value-grounding decision: does ACTUALone-step successor information improve held-out prediction of eventual105success over currentstate/directQ/knownmotion-time controls? This truth-based upperbound is distinct from reopening failed physical predictors or feeding them to an actor. Only useful task-value lookahead would justify a new corresponding Cm→successor→V design.
