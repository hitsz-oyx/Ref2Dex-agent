# Return-corrected physical-model gradients: established method, open utility

Primary sources consulted2October2026:

- [Q-Prop, Gu et al., ICLR2017](https://arxiv.org/pdf/1611.02247).
  Retrieved13-page PDF; sections3.1--3.2 provide the Taylor control-variate
  estimator, analytic correction and distinction between surrogate advantage
  variance and parameter-gradient variance. OpenReview mirror instead returned
  a browser-verification page; no access to that mirror claimed.
- [The Mirage of Action-Dependent Baselines, Tucker et al., ICML2018](https://proceedings.mlr.press/v80/tucker18a.html),
  [author institutional abstract](https://research.google/pubs/the-mirage-of-action-dependent-baselines-in-reinforcement-learning/).
  These primary abstracts report that variance benefits did not hold in their
  tested domains and that implementation bias explained apparent gains.
- [Stein action-dependent control variates, Liu et al.](https://arxiv.org/abs/1710.11198).
  Primary abstract only used: broader action-dependent baselines are prior art.

Therefore merely putting our Cm->successor->V critic inside Q-Prop is not a
new algorithm or a journal-level claim. Our negative offline actor result
does not establish that its derivative is useful as a control variate either.
No claim of new unbiased-gradient theory or guaranteed variance reduction.

Candidate experiment algebra (our Gaussian held-option specialization): for
raw a=mu(x)+epsilon, epsilon~N(0,sigma^2 I), fixed sigma, let c(x) be shared
action-independent baseline and b(x) a critic action derivative computed BEFORE
this episode's action/return. Let J=derivative of actor mean wrt parameters.
Measured terminal return R is the unchanged physical105 bool. Compare

`g0 = J.T @ ((R-c)*epsilon/sigma^2)`

`gb = J.T @ (((R-c-b.dot(epsilon))*epsilon/sigma^2)+b)`.

Given x and models fitted without this episode, Gaussian E[epsilon]=0 and
E[epsilon epsilon.T]=sigma^2 I make E[gb-g0|x]=0. The b term is necessary;
subtracting the action-dependent term alone is biased. Keep b and c detached,
fixed coefficient1 and fixed covariance. Native tanh is part of the task
return, not an excuse to change the raw Gaussian score. No own-return fitting
of b/c/coefficient and no action clipping before the native tanh.

This identity says nothing about benefit: poor derivatives can INCREASE noise.
The next cheapest Decision is fresh on-policy measured returns and COMPLETE
actor-parameter gradients, comparing Cm, physical-action-removed and direct-Q
derivatives against a common baseline. Pooled advantage/score energy is not
sufficient. If promising, actual matched learning and independent evaluations
are still required, followed by distinct-method work; this mechanism alone
does not complete either the policy-utility or publication objective.
