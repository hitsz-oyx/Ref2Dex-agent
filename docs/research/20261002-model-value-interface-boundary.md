# Physical successor → task value: existing-method boundary

Primary sources checked2October2026; abstracts/publisher descriptions, not a full implementation reproduction. This note serves the next-route decision, not an empirical result. Single-session research, no delegation.

[MVE, Feinberg et al.2018](https://arxiv.org/abs/1803.00101) uses learned dynamics for short, fixed-depth imagined trajectories and value expansion. Thus a generic Cm successor followed by a critic is established methodology; it is not novel merely because the model is called Cm.

[TD-MPC, Hansen et al.2022, ICML](https://proceedings.mlr.press/v162/hansen22a.html) jointly learns task-oriented latent dynamics and terminal value, combines short local planning with long-term return. Task-relevant latent prediction and terminal-value planning also require a substantive departure, not relabeling.

[Q-Prop, Gu et al.2017, authors' research page](https://research.google/pubs/q-prop-sample-efficient-policy-gradient-with-an-off-policy-critic/) uses a critic Taylor approximation as a policy-gradient control variate. A generic model-derived gradient control variate alone is similarly insufficient as novelty.

Local inference: this branch's failed auxiliary-critic representation does NOT experimentally test a full Cm(s,a)→predicted successor→task-value chain. Conversely, replacing real next states for executed actions with model predictions is not an automatic improvement. Cm's potentially useful role concerns unexecuted actions and limited-policy-data value learning. Physical height change must not be substituted for long-term105grasp value.

Any future route must specify the complete successor representation, causal actuator execution, genuinely task-relevant value target, direct Q/state-only/strong fixed controls, and an independently evaluated learned actor. We still need a distinctive method and positive matched policy-training evidence; these sources cannot supply either for this repository. Do not reopen closed one-tick/four-tick predictor recipes as a generic MVE renaming.
