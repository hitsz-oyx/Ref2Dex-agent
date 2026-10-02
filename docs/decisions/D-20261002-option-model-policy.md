# Execute a task-value model chain in trained option policies

Question: can a short physical successor actually change policy learning for
joint held options, beyond state-only dynamics and direct task-Q learning?

Evidence: many immediate/coherent physical forecasts and old auxiliary-critic
training designs fail. These do not implement this physical-successor plus
held-option continuation-value chain. Actual-state randomization remains
available despite failed exact replay. Random option mean is worse than P0,
so an oracle maximum cannot establish benefit. Use trained actors' prospective
native outcomes as the decision, not another predictor-error screen.

Action: prospectively frozen new603/604FIT and611/612EVAL, no old601fit/test.
Cm predicts8-step observed physical state; continuation value conditions on
the held option; bounded regularized offline policy optimization trains actual
actors. Matched action-removed dynamics and same-observation direct-Q actor
control alternative explanations. Original spaced native scene and physical105
remain unchanged. Frozen-model exploitation is constrained, not assumed solved.

Cost<=1200s/2GiB, one idle GPU,3072trajectories/9000neural optimizer steps.
Primary actual success gates alone choose the next route; no intermediate
forecast/loss-based selection. Positive permits formal matched Validation and
novelty work; negative ends this exact recipe without local scans. Generic
offline model-based policy optimization is prior art; Mission/claim/resources
unchanged, no additional external authorization needed.
