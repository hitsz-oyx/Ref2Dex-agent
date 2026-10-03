# P-20261003-cm-uncertainty-acquisition

Family: Cm uncertainty-guided data responsibility
Type: Held-only decision Probe
Status: COMPLETED — `PROMISING`

## Question

Does Cm ensemble disagreement identify the transitions where targeted consequence data
would be more useful than ordinary uniform data?

## Contract

On the fixed 209,788-row held split, uncertainty was the ensemble standard deviation
over object translation/velocity/contact/reward/terminal predictions for the actual
action. The orientation head was excluded. We compared uncertainty with realized
normalized physical prediction error and absolute direct-Q return residual. No Cm fit,
action, reward, or policy was changed.

## Result

The top-disagreement 20% had `7.03x` the physical error of the bottom 50%, carried
`32.5%` of absolute direct-Q residual mass, and covered all `384` held episodes.
Spearman correlations were `0.535` with physical error and `0.058` with absolute
direct-Q residual. All predeclared data-responsibility gates passed, so the screen is
`PROMISING` as an acquisition signal only.

The result does not establish policy utility. It authorizes one fixed high-uncertainty
fit/retraining Probe; no ordinary data expansion, threshold scan, or policy training is
authorized until that targeted model screen is complete.

Evidence: [results](P-20261003-cm-uncertainty-acquisition-results.json), [Decision Memo](../../decisions/D-20261003-cm-uncertainty-acquisition.md).
