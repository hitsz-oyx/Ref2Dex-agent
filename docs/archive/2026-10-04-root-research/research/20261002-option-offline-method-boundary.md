# Offline option model/value policy optimization: prior-art boundary

This session uses the research skill directly, under the user's explicit
single-session instructions; its background-agent instruction is overridden.

Learning a dynamics model from logged data and optimizing a policy through
model predictions is established offline model-based RL. MOPO explicitly
addresses offline distribution shift with a penalty tied to model uncertainty;
[MOPO primary publication](https://proceedings.nips.cc/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html).
COMBO learns conservative values using offline and model-generated data and
regularizes out-of-support model-rollout state/action tuples without explicit
uncertainty estimation;
[COMBO primary publication](https://proceedings.neurips.cc/paper/2021/hash/f29a179746902e331572c483c45e5086-Abstract.html).
Only the primary publisher abstracts/summaries are reviewed here; these are
not reproductions or full-text algorithm audits.

Our bounded raw option and mean-square penalty constrain proposals but do not
provide either paper's conservatism guarantee. The eight-step Cm->continuation
V composition is not automatically distinctive, and a positive Probe would
not establish novelty or Validation. Its immediate purpose is an actual
same-data policy-learning intervention, beyond earlier prediction-only screens.
The continuing option must be an explicit V input; otherwise different future
controllers would be mixed into an ill-specified value target.
