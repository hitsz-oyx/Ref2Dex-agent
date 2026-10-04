# Response-field candidate: established foundations, utility unproved

Primary sources checked2026-10-02:

* [Suh et al., ICML2022](https://proceedings.mlr.press/v162/suh22b.html)
  study first/zeroth-order policy estimators in stiff/discontinuous physical
  systems. Gaussian score estimators, state baselines and smoothing are established;
  the proposed response moment is not a novel score identity.
* [IAEM, AAAI2022](https://ojs.aaai.org/index.php/AAAI/article/view/20913)
  learns action effects from residual state representations, contrastive
  invariance and adaptive treatment of noninvariant cases. Action-effect features
  helping policy learning, including departures from invariance, are occupied.
* [SALE/TD7, NeurIPS2023](https://arxiv.org/abs/2306.02451)
  uses learned state-action interaction embeddings for low-level continuous
  control. Generic dynamics representations attached to policy learning are
  not a distinct contribution. This note uses its abstract only, not an asserted
  architecture-level comparison; publisher PDF fetch failed due to file size.

Candidate here: learn a current-state field of randomized request-to-object
response moments; distinguish response from baseline drift and consume the
field before choosing actions. The exact physical moment, command projection,
matched actor interface and retention task could define an empirical method,
but none establishes novelty alone. Population linear-response regression and
Gaussian moment regression can have the same optimum; do not claim otherwise.
First decide whether state conditioning beats a strong motion/phase response
mean on fresh identically distributed data. Formal method comparisons and
matched positive policy-training utility remain required after a positive Probe.
