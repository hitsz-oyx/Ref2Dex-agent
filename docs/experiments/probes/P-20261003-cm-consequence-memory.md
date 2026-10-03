# P-20261003-cm-consequence-memory

Family: Cm observation responsibility
Type: Held-only decision Probe
Status: COMPLETED — `UNPROMISING`

## Question

Can the prediction error of Cm's previous real action provide a useful regime memory
for the next decision, without turning Cm into a success predictor?

## Contract

The frozen physical ensemble predicted one-step object translation/velocity, contact,
events, reward and terminal consequences. At row `t`, the candidate feature was only
the observed-minus-predicted consequence of row `t-1`, its ensemble disagreement and a
valid predecessor mask. The current row's future fields were not used. Direct-Q and Cm
were frozen; only a small fit-split residual value head was trained. A shuffled-memory
control used the same architecture and update budget.

## Result

The full screen used `120,000` fit rows and `209,788` held rows from `1,536/384`
episode-disjoint fit/held episodes. Valid memory coverage was `99.82%`. Direct-Q held
RMSE/MAE/row Spearman/episode Spearman were
`26.833/9.108/0.589/0.743`. Real consequence memory produced
`27.072/9.335/0.582/0.769`; shuffled memory produced
`27.670/10.075/0.558/0.772`.

The real residual improved episode Spearman by `0.026` and beat the shuffled control in
RMSE by `0.598`, but worsened RMSE by `0.238` against direct-Q. The predeclared RMSE
improvement gate therefore failed. Close this memory contract; do not run native
collection, policy training, or threshold/seed/horizon scans. The episode-level change
is retained as a mixed diagnostic clue, not Cm policy utility.

Evidence: [results](P-20261003-cm-consequence-memory-results.json), [Decision Memo](../../decisions/D-20261003-cm-consequence-memory.md).
