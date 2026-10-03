# Probe: fixed Cm fit on native uncertainty-targeted transitions

Date: 2026-10-03  
Experiment ID: `P-20261003-cm-targeted-transition-fit`  
Status: `UNPROMISING`

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

## Result

The fixed fit used all `36` targeted first-step transitions for `600` updates, adapting only
the final output projection. Training loss fell from `0.579215` to `0.111830`, but the held
physical RMSE worsened from `0.482034` to `0.565057` overall (`+17.2%`) and from `0.905606`
to `1.000025` in the high-uncertainty subset (`+10.4%`). The conservative value-target RMSE
worsened from `26.622387` to `26.713272`; episode Spearman changed by `+0.00826`.

The Probe is `UNPROMISING`. The actual targeted transitions are useful for locating a
disagreement regime, but this small fit over-specializes and transfers poorly to the held
distribution. Close targeted fitting and do not start policy training, ordinary data
expansion, or threshold/seed/horizon scans.
