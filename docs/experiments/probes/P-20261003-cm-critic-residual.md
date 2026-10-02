# P-20261003-cm-critic-residual

Family: Cm decision interface
Type: Offline critic sufficiency Decision Probe
Status: COMPLETED — `UNCLEAR`, close this linear residual interface

This Probe tested whether Cm's physical consequences contain task-value information
that direct-Q misses, without putting a predicted state into an actor. The frozen
direct-Q was the baseline. A Ridge residual head was fit on the fit split to predict
realized complete return minus direct-Q, then evaluated on the held split. The Cm
head received only frozen ensemble predictions of object delta, velocity, contact,
reward, terminal, and their uncertainty, together with current state/action. No
future state, success label, actor update, or policy execution was used.

Across 831,811 fit and 209,788 held transitions (1,536 and 384 episodes), direct-Q
achieved held RMSE `26.833`, row Spearman `0.589`, and episode Spearman `0.743`.
Adding a current-state/action residual worsened all four metrics. Adding Cm
physical residual features improved episode Spearman to `0.761` (`+0.018`), but
worsened RMSE to `27.050`, MAE to `9.589`, and row Spearman to `0.586`. The mixed
result is not a clear held value improvement, and it does not repair the negative
native action-utility screens.

Close this exact linear critic-residual interface. Retain the result as evidence
that Cm has a weak episode-level value-ranking clue, but do not train a policy or
claim Cm utility from it. Any continuation must use a new representation or a
decision target with a predeclared held criterion.

Full metrics and hashes are in [`P-20261003-cm-critic-residual-results.json`](P-20261003-cm-critic-residual-results.json).
