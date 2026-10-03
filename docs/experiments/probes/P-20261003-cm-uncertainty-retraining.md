# Probe: fixed high-uncertainty Cm retraining

Date: 2026-10-03  
Experiment ID: `P-20261003-cm-uncertainty-retraining`  
Status: `UNPROMISING`

## Decision question

Does allocating Cm updates to the frozen disagreement-defined high-uncertainty regime
make its physical consequences useful for the conservative one-step task-value target?

## Contract

The fit split was fixed to 120,000 rows. The frozen three-member Cm ensemble selected its
top 20% disagreement rows (24,000 rows), and each member received exactly 600 updates on
that subset. The held split was untouched. Object orientation stayed excluded from the
physical target; direct-Q, actor, reward and success definitions were unchanged.

The predeclared gates were: at least 10% improvement in held top-regime physical RMSE,
overall physical RMSE no worse than 2%, at least 0.5 RMSE improvement on the conservative
one-step value target, and episode Spearman loss no larger than 0.01. A failure closes the
retraining route and authorizes no policy run.

## Result

The fixed high-uncertainty fit improved held physical RMSE from `0.905606` to `0.859734`
in the selected regime, a `5.1%` improvement that missed the 10% gate. Overall physical
RMSE improved from `0.482034` to `0.466745` (`3.2%`), while the conservative value-target
RMSE changed from `26.622387` to `26.632624` and MAE from `8.872545` to `8.974402`.
Episode Spearman increased `0.00678`; it did not compensate for the value RMSE failure.

The Probe is therefore `UNPROMISING`: disagreement identifies a real physical-error
regime, but fixed reweighting of existing rows does not convert that signal into a better
task-value target. No policy training, ordinary data expansion, or threshold/seed/horizon
scan follows. The acquisition result remains a data-local lead; a future route would need
actual targeted transitions or a new decision responsibility.

Evidence: [results JSON](P-20261003-cm-uncertainty-retraining-results.json),
[Decision Memo](../../decisions/D-20261003-cm-uncertainty-retraining.md).
