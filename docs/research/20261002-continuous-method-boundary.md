# Continuous critic and action-likelihood methodological boundary

Read full primary texts,2October2026; distinction from generic work is NOT
established. This record does not change the frozen continuous Probe recipe.

[UNREAL, Jaderberg et al.](https://arxiv.org/html/1611.05397), sections3--4:
auxiliary prediction/control losses shape shared representations alongside
extrinsic actor-critic learning; section4compares input-change prediction with
auxiliary control. Inference for this branch: attaching object-transition
prediction to a value encoder is not novel merely because the target is physical.
Our separate actor encoder and current command-conditioned critic differ in
implementation, not demonstrated conceptual contribution. No independent novelty
claim or copied performance number is used.

[CAPG, Fujita and Maeda, ICML2018](https://proceedings.mlr.press/v80/fujita18a/fujita18a.pdf),
sections3.1--3.3and3.5: clipping maps request tails to boundary mass; tail-CDF
scores provide an unbiased lower-variance estimator under stated assumptions.
Section3allows state-dependent bounds. Inference: a proposed Gaussian-tail or
Rao--Blackwell correction for this branch's native projection would substantially
reproduce CAPG, even with monotone tanh and dependent-finger reconstruction.
The current frozen Probe intentionally uses REQUEST likelihood and Gaussian
request entropy; it does not claim executed-target density/entropy or implement
CAPG. No recipe change is made midexperiment.

These papers do not settle the usefulness of the actual matched continuous
native manipulation comparison. One-step forecasting loss, bounds diagnostics
and engineering audits cannot substitute for final prospective policy utility.
A positive result would require distinguishing technical substance, stronger
matched baselines, training-seed Validation, native task/object generalization,
and sensory/hardware transfer before the journal objective could be achieved.
