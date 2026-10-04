# Physical predictions as control variates: primary-source check

Research skill applied within the authorized single-session workflow. Source
checks below are primary abstracts/proceedings, not a novelty verification of
our future implementation.

- [Q-Prop, Gu et al., ICLR2017](https://arxiv.org/abs/1611.02247v3)
  uses an off-policy critic's local Taylor approximation as a policy-gradient
  control variate; conservative and aggressive variants adapt its use.
- [Expected Policy Gradients, Ciosek and Whiteson, JMLR2020](https://arxiv.org/abs/1801.03326v2)
  integrates or sums over actions, including a discrete softmax formulation.
  Exact action summation itself is established, not our contribution.
- [From Importance Sampling to Doubly Robust Policy Gradient, Huang and Jiang,
  ICML2020](https://proceedings.mlr.press/v119/huang20b.html)
  derives flexible gradient estimators from doubly robust off-policy evaluation
  and analyzes their variance. Model-corrected policy gradients are established.

Our bounded question is empirical: can a fixed short-horizon physical-support
forecast serve as a useful control variate for the LONG physical105task reward?
The support forecast need not be an accurate long-term value function for the
finite-action correction identity below to hold. This is our algebraic
specialization of established control-variate principles, not a new theorem.

For a fixed current state, stochastic policy pi(a), known behavior p(a)>0,
score h(a)=grad log pi(a), and arbitrary fixed q(a), define

`g = pi(a)/p(a) * h(a) * (R-q(a)) + sum_b pi(b)*h(b)*q(b)`.

Taking expectation over the assigned behavior action cancels the q terms,
leaving `sum_a grad pi(a) * E[R|a]`. Thus misspecified q does not bias this
conditional contextual-bandit gradient, provided the correction is exact,
q is held fixed for this gradient, assignment probabilities are correct and
current state precedes the intervention. This does NOT automatically extend
to changing state occupancy, clipped-PPO gradients, action-dependent estimated
scales, or unobserved continuation changes.

This permits a cheap new reused-data mechanism test after the failed forecasting
gate, without retraining the failed Cm, relabeling the old result, or directly
turning a short-horizon forecast into reward. Lower prediction loss alone does
not guarantee lower gradient noise. Any eventual claim about sample efficiency
still requires new matched policy-training and independent physical tests.
