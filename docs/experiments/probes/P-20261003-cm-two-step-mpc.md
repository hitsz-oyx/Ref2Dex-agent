# P-20261003-cm-two-step-mpc

Family: Cm model-predictive planning
Type: Held-only decision Probe
Status: COMPLETED — `UNPROMISING`

## Question

Can Cm improve action decisions when it recursively imagines two consecutive local
actions, instead of scoring only one action or shaping a critic target?

## Contract

For each held transition, the frozen Cm ensemble predicted two physical transitions.
The second action was selected from the same bounded local candidate panel around the
imagined first action, and the frozen direct-Q supplied the tail value. Ensemble scores
used mean minus standard deviation. Observed hand q/dq and object orientation were
preserved; Cm supplied object translation/velocity/contact/events. No actor, success
predictor, policy, or native simulator was changed.

The predeclared 512-row screen required RMSE improvement of at least `0.5`, episode
Spearman loss no larger than `0.01`, and first-action argmax changes between `5%` and
`60%` before native collection would be considered.

## Result

On 512 held rows from 384 episodes, direct-Q scored RMSE `31.029`, MAE `11.334`, row
Spearman `0.553`, and episode Spearman `0.554`. One-step MVE scored RMSE `31.189`; the
two-step MPC scored RMSE `34.817`, MAE `11.274`, row Spearman `0.556`, and episode
Spearman `0.563`. Thus two-step RMSE worsened by `3.788` and the first-action argmax
changed on `82.8%` of rows, beyond the predeclared reliable-change range.

Close this planning contract. Do not run native collection, policy training, horizon
scans, action-panel scans, or more model-based critic variants. This result concerns
the recursive planner only; Cm remains a real one-step physical consequence predictor.

Evidence: [results](P-20261003-cm-two-step-mpc-results.json), [decision memo](../../decisions/D-20261003-cm-two-step-mpc.md).
