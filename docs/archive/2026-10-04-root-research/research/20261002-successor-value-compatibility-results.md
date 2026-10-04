# Actual eight-step successor carries value that the learned composition misses

Reused-data Decision `P-20261002-successor-value-compatibility`, designe136c3c,
implementation9a0f873, r1 COMPLETED/PROMISING. All768previous618episodes,
fixed603/604models; zero new trajectories, fitting or actor updates.

| Fixed prediction | Brier |
| --- | ---: |
| Direct Q(current,held option) |.1224600595|
| V(actual8-step successor,held option), oracle only |.0844091163|
| V(learned Cm successor,held option) |.1204715339|
| V(action-removed successor,held option) |.1359789609|

Oracle improves directQ by31.0713%; paired95%bootstrap interval oracle-minus-Q
[-.05868426,-.01808177]. Learned Cm excess overoracle42.7234%; interval
Cm-minus-oracle[.01707893,.05645126]. All four predeclared gates pass.
Useful for choosing a DIFFERENT physical representation, not a policy result.
Actual future contains execution/noise outcomes unavailable to a current-state
model; its advantage is not a deployable information guarantee. A learned
conditional mean, Jensen explanation and calibrated uncertainty are unproven.
Single previously used native panel, not an independent Validation dataset.

Current/future rawSDK features reconstruct exactly0error, publicfutureplan
and unchanged105mesh labels independently checked; full four NumPy forwards
<=1.098e-6, exact Brier/bootstrap/gates after checking recorded float32 outputs.
All inputs unchanged;79.290s/9644276bytes<=180s/32MiB, GPU6 for fixedmodel
inference, CPU audits, all own PIDs terminal. PredictionsSHA256:
`1e3aad41aa420b2cec1121d022e4a8cc5161bcf3c5510603bc2c9f54c0301d15`.

Decision: preserve complete observed successor tuples in a learned atomic
physical transition model, optimize actors using the expected continuation
value across that support, and compare actual deployed actors with a matched
physical-action-removed model, unchanged P0 and the strong retained direct-Q
actor. This does not reopen the failed deterministic/CV recipes or prove why
they failed. Generic kernel models/uncertainty/model-based RL are prior art.
