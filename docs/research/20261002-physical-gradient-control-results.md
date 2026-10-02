# Fresh full actor-gradient control variates: negative

Design8dc79bd, implementationf0be3ca. Experiment
`P-20261002-physical-gradient-control`, r1 COMPLETED/UNPROMISING.
New618native768trajectories/202ticks,192P0reference and576Gaussian N(0,I)
held options from the SAME zero-mean actor. Private independent per-env draws
and placements, all three Gaussian arms pooled; no antithetic/replay assumption.
No new model/actor optimizer step; previous6000model/3000offline-actor steps
reported. Those actor means are not used in this diagnostic.

Models from603/604 are fixed before all fresh actions/returns. Actual terminal
physical105 return, public plan/clock and CURRENT152native state only; no actual
future physics enters critic derivatives. Shared direct-Q(x,0) state baseline,
fixed covarianceI and coefficient1. The analytic +b correction remains present.
This is a complete14732-parameter actor-gradient comparison at the actual
scratch3555zero-head initialization, not advantage/score/output energy alone.

| Estimator | Complete parameter-gradient covariance trace |
| --- | ---: |
| Common state baseline |8.2512754004|
| Physical Cm derivative control variate |32.5815904520|
| Action-removed physical derivative control |11.9448314244|
| Direct task-Q derivative control |14.0499617525|

Paired stratified1000-bootstrap95% intervals, Cm minus control:
baseline[6.696385,48.630570], off[3.337401,44.537376],
directQ[.754264,43.312522]. Both mandatory gates fail: no10%reduction over
all three and no negative upper intervals. Cm trace is3.9487xbaseline,
2.7277xoff and2.3190xdirectQ in this fixed first-update distribution.
This does not prove universal model harm, estimator bias or actual learning
harm; it rejects investing in THIS fixed derivative-control recipe. The algebra
and tiny exact-moment smoke establish the correction identity, not empirical
conditional unbiasedness or a guarantee of useful variance reduction.

All768native current/P0/private RNG/options/PD/full-mesh audits pass; native
mesh<=2.322e-7m/P0forward<=1.755e-7. Native arm successes[55,49,45,55]are
behavior context, not treatment effects: three stochastic arms have identical
policies. New gradient source inputs independently reconstruct exactly0error;
critic forwards<=1.200e-6 and raw reverse derivatives<=7.643e-6.
Every actual Torch actor-parameter gradient checked independently: hidden
blocks exactlyzero, all final-head gradients<=2.094e-6. Only after independent
checks are the recorded float32 critic/vector blocks reused for exact variance
and bootstrap reconstruction. Complete zero hidden blocks are verified,
so numerical compression does not change the full-gradient metric.

Cost206.507s/391985146bytes<=300s/512MiB, fresh GPU6 for native/gradient
computations and CPU independent audits. All protected original675inputs,
fixed models and sources unchanged, all own PIDs terminal. Gradients SHA256:
`79c0c7525086e436008477f3eaa2c2042dedff1e7f0b2854220d0530f228ceb9`.

No coefficient, sigma, baseline, model, width, seed or threshold scans; no
actual corrected actor training launched from this failed gate. Generic
Q-Prop/Stein machinery is established, and this track has no journal-level
Cm utility. Revisit physical-state distributions and task compatibility
before fitting another deterministic predictor or derivative method.
