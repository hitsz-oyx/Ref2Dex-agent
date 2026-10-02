# P-20261003-cm-direct-q-mve-object-projection

Family: Cm decision interface
Type: Decision Probe
Status: COMPLETED — `UNCLEAR` support boundary, with a negative native utility screen

This Probe implements the requested decision-time contract: Cm predicts only object
translation/velocity/contact/events; the current hand configuration and object
orientation are retained; and the one-step score is `local reward + gamma *
direct-Q(predicted state, candidate action)`. The frozen candidate panel includes
Cup, direct-Q, the prior Cm value interface, and a uniform random arm. No actor,
model, or PPO update is performed.

On the existing 209,788-row held split, direct-Q continuation gave object-projected
MVE RMSE 26.60 versus direct-Q 26.83, row Spearman 0.591 versus 0.589, and episode
Spearman 0.741 versus 0.743. A fresh native panel produced 119 complete windows;
MVE changed 29 actions. Relative to Cup, its cluster-bootstrap lower90 bounds were
-34.02 mm for last-three-step minimum height, -0.173 for local reward, -0.215 for
contact fraction, and -0.251 for clearance. On the 22 random-arm rows, its height
and reward ranking Spearman were 0.360 and 0.503, below direct-Q at 0.417 and 0.531.

The action-ranking, changed-action utility, and support gates therefore do not pass.
Close this exact direct-Q MVE interface; do not start PPO, lower the gates, or expand
ordinary Cm data. The offline target-quality change remains a critic-only clue and
does not establish policy utility. Full values are in
[`P-20261003-cm-direct-q-mve-object-projection-results.json`](P-20261003-cm-direct-q-mve-object-projection-results.json).
