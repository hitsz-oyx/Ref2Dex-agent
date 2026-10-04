# Disturbance recovery: occupied territory

Primary abstracts inspected on2October2026; these constrain novelty, not our
empirical claims. Their hardware / other-platform results are not our results.

- [Nagabandi et al., Deep Dynamics Models for Learning Dexterous Manipulation,
  CoRL/PMLR100,2020](https://proceedings.mlr.press/v100/nagabandi20a.html) already
  combines learned dynamics with online planning for complex dexterous tasks and
  demonstrates real-world manipulation. Learned dynamics plus MPC is established.
- [Jiang et al., Robust Model-Based In-Hand Manipulation with Integrated Real-Time
  Motion-Contact Planning and Tracking,2025](https://arxiv.org/abs/2505.04978)
  combines contact-implicit high-level planning with low-level force/motion
  tracking and reports robustness to external disturbances on real robots.
  Disturbance recovery or joint contact/motion planning alone cannot be our novelty.
- [Suh et al., Dexterous contact-rich manipulation via the contact trust region,
  IJRR2026](https://journals.sagepub.com/doi/10.1177/02783649251398875) models unilateral
  contact through a contact trust region and connects local MPC plans to global
  plans, including simulation/hardware evaluations. Contact-mode-aware planning
  is also established.

Our next finite-load Probe is task calibration only. A distinctive method still
needs a precise technical advance and independent matched policy-training utility,
not a new name for support forecasting, a selected force dose, or another failure.
