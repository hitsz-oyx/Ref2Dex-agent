# Literature and repository evidence for contact-response research

Read on 2026-10-01. This is a focused novelty screen, not an exhaustive survey.

## Primary sources

1. [Scaling Cross-Embodiment World Models for Dexterous Manipulation](https://arxiv.org/html/2511.01177v1),
   He et al., 2025: already uses shared hand/object particles and particle-displacement
   actions, with dynamics learning and planning. Point-flow cross-hand modeling
   alone is not an original contribution for this project.
2. [Dexplore](https://arxiv.org/abs/2509.09671), Zheng et al., 2025:
   reference-scoped exploration provides the existing manipulation substrate.
   Our pilot uses the repository's self-trained actor, not released official weights.
3. [Value-Aware Loss Function for Model-based Reinforcement Learning](https://proceedings.mlr.press/v54/farahmand17a.html),
   Farahmand, Barreto and Nikovski, 2017: model learning objectives can depend on
   the downstream value function. Value awareness is not a new general idea.
4. [AD-WM: Action-Discriminative World Models for Counterfactual MPC](https://arxiv.org/abs/2609.30264v2),
   Qiu et al., September 2026: predictor-level action recovery preserves action
   information, and the paper studies prediction metrics versus control outcomes.
   Generic action-discrimination or the observation that prediction accuracy
   does not imply policy utility would overlap directly with this work.
5. [How Should World Models Be Evaluated for Embodied Decision-Making?](https://arxiv.org/abs/2606.15032),
   2026 position paper: advocates evaluation under interventions and closed-loop
   decision making. A general counterfactual benchmark claim needs more than
   renaming this established evaluation principle.

## Local evidence and interpretation

- Formal effect-rank validation in `docs/experiments/validations/VAL-20260923-CM-EFFECT-PPO.md`
  does not support the joint positive policy/action-alignment claim.
- `ObjectInteractionCm`'s September 12 geometric-baseline audit found that mean
  hand displacement accounts for most of the full model's error reduction
  relative to zero flow. It still found an average full-model advantage; this
  does not establish that the model is useless.
- Read-only HD02 r4 results have 34/384 versus 35/384 closed-loop successes but
  63 discordant success labels. The near-identical marginal rates hide substantial
  variation in which episodes succeed. Saved-action replay behaved differently;
  it is a physical diagnostic, not a replacement for closed-loop evaluation.

## Candidate ideas and selection

1. Larger mixed-source Cm pretraining: high cost and substantial overlap with
   cross-embodiment particle world models; no evidence this fixes policy utility.
2. Generic value-aware/action-discriminative loss: plausible but crowded and
   presupposes that candidate physical contrasts can be measured reliably.
3. Contact-conditioned differential action effects with explicit noise resolution:
   chosen for the first pilot. A fixed pulse/repeat diagnostic can cheaply test
   whether informative local contrasts exist before training a new model.

Potential research question: can a model learn and use the smallest reliable
contact-dependent physical response to an action change, while avoiding long-horizon
noise amplification? A possible method uses the contrast
`f(history, action) - f(history, reference_action)` and contact-conditioned
prediction horizons, then distills improvements into a normal actor. This is
an untested proposal. It is not claimed to be the first contrastive dynamics model.

## Requirements for a journal-level claim

The pilot cannot support a top-journal paper on its own. A credible submission
would need task-level causal improvement under matched interaction/compute budgets,
multiple independent training seeds, independent object/task splits, realistic
contact labels, an unseen hand or other consequential transfer setting, hardware
evidence where feasible, and strong geometric/model-free/model-based baselines.
Metrics must include post-lift retention and drops; short held-lift or flow error
alone is insufficient. These requirements describe missing evidence, not results.
