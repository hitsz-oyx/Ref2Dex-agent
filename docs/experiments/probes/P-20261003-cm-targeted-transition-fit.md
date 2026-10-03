# Probe: fixed Cm fit on native uncertainty-targeted transitions

Date: 2026-10-03  
Experiment ID: `P-20261003-cm-targeted-transition-fit`  
Status: `AUTHORIZED`

HF30 passed the native acquisition screen and authorized exactly this follow-up. The
Probe will use only the `36` targeted first-step transitions from the completed native
record, keep direct-Q, actor, reward definition and held rows fixed, and update the frozen
Cm dynamics ensemble once. The last output layer is adapted with a deterministic 600-step
fit; no threshold, seed, horizon or model-weight scan is allowed.

The held screen will compare the original and adapted ensemble on the existing 209,788-row
held split. The predeclared gates are top-uncertainty physical RMSE improvement of at least
10%, overall physical RMSE no worse than 2%, conservative value-target RMSE improvement of
at least `0.5`, and episode-Spearman loss no larger than `0.01`. A failure closes targeted
transition fitting and authorizes no policy training. A pass authorizes at most one fixed
matched policy Probe.
