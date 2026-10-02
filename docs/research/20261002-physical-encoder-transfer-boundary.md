# Physical encoder transfer: primary-source boundary

Read Fan et al., [Pretraining in Actor-Critic Reinforcement Learning for
Locomotion](https://arxiv.org/html/2510.12363v4), RA-L2026 author manuscript,
sectionsIV-D/IV-E andV-D. They pretrain an inverse-dynamics backbone, replace
prediction-specific modules with task modules, and initialize both actor and
critic. All transferred parameters remain trainable. Critic-only and actor-only
ablations are reported. Generic dynamics weight transfer into RL is prior art.
Our forward action-conditioned Q encoder with measured terminal-return fitting
is an implementation choice, not a novel method established by that distinction.
We do not reproduce their inverse-dynamics architecture or locomotion results.

Also verified [Du and Narasimhan, Task-Agnostic Dynamics Priors,
ICML2019](https://proceedings.mlr.press/v97/du19e.html), proceedings abstract:
physical-video predictor pretraining and target dynamics fine-tuning are older
related work. No full-text theorem or same algorithm claim from the abstract.

A task-critic gain would justify validation, not journal readiness. Extra physical
pretraining data/compute must be explicit; current source includes measured full
episodes, so no total environment-sample saving follows from weight reuse.
