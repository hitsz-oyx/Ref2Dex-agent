# Observed successor law: actual policy-learning result

Decision Probe P-20261002-empirical-successor-policy, design9b84c45,
implementation6efc6c0, run-r1 COMPLETED / UNPROMISING. ReusesONLY1536FIT603/604
trajectories, existing fixed continuation V and direct-Q actor. New physical
models3000optimizer steps, new actual actors2000steps; source6000model/3000actor
steps separately retained. Reused direct-Q actor1000steps are not repeated.
Fresh EVAL623/624 contributes1536native trajectories,768environments/202ticks
per panel. Independent randomized placements/arms, no same-state replay claim.

| Method | Physical105 /384 | Rate | EVAL623 /192 | EVAL624 /192 |
| --- | ---: | ---: | ---: | ---: |
| P0 |126|32.81%|68|58|
| Cm conditional observed-successor law |188|48.96%|89|99|
| Action-removed physical law |188|48.96%|94|94|
| Reused direct-Q actor |222|57.81%|107|115|

| Motion | P0 /128 | Cm /128 | Physical-off /128 | Direct-Q /128 |
| --- | ---: | ---: | ---: | ---: |
|0|0|0|0|0|
|1|105|106|110|111|
|2|21|82|78|111|

Frozen >=5pp over EACHcontrol: FAIL. Each-seed noninferiority toALLcontrols:
FAIL. Motion1 loss<=5pp against P0: PASS. Primary **UNPROMISING**. Cm beats P0
by16.15pp descriptively, ties physical-off and trails direct-Q by8.85pp.
This does not demonstrate action-conditioned physical-model policy utility.
One shared FIT and two evaluation seeds are not formal Validation.

The model predicts probabilities over all1536observed physical131-feature
successor tuples, with own episode excluded during physical and actor fits.
Every atom receives the query's public future21plan/clock; V is evaluated on
EACHtuple before probability averaging. These tuples are observed features,
not guaranteed physically feasible future states conditional on a new query.
Neither calibrated uncertainty nor a Jensen explanation is established.
Learned actors alone deploy; no model, support lookup or future input at inference.
Physical105 remains30mmroot rise/20mmfullmeshclearance on ALL75syntheticplateau
plus30drop-check ticks. Original90syntheticplateau and privileged state explicit.

All1536FIT raw current/successor SDK inputs independently reconstructed with
maximum discrepancy0. Independent physical kernel error<=9.10e-13; encoder
forwards<=6.49e-6, probability<=6.07e-7, logprobability<=9.69e-5; literal separate
full-input expected-value forwards<=1.77e-7 and value-of-mean<=4.55e-7.
Recorded float32 probabilities/logweights reused AFTERindependent checks for
composition, as prospectively specified. Final fullFIT actor checks<=4.51e-7.
Direct-Q weights and actor initial parameters bitwise source; actual nonzero
first gradients and changed parameters recorded. No independent optimizer replay.

Both native audits pass: all actual actor observations reconstructed exactly;
all deployed NumPyactor outputs<=3.05e-7; fullmesh clearance<=4.69e-7m,
PD/input/control/task105 labels retained. All protected source hashes unchanged,
all owned parent/child PIDs absent at terminal.417.390s/591055743bytes<=1200s/1GiB.
Result SHA2563f585e4aa6ae03094abd264a1909df7f20529914ef6c119287234f2d750311dc.
Actor SHA2561f51478194adf7124bf440df1c2437663c5edf5a381422f0507ac41f31d18298.

Checkpoint limitation: physical_update1500.pt contains final physical weights,
not the physical optimizer or generator. Actor_step0500/1000.pt contains actual
actor weights, optimizers and private batch-generator states. The card's earlier
collective checkpoint wording was broader than that implementation; no optimizer
replay/resume ability is inferred for the completed physical phase. No scientific
updates or evaluation were repeated, checkpoints or tolerances changed.

Close this exact atomic/kernel expected-value actor recipe without kernel,
temperature, support, width, epoch, seed or penalty scans. Oracle successor
headroom remains a diagnostic, not a deployable result. Next assess transfer of
already learned physical encoders into a critic trained on measured terminal
returns, instead of optimizing predicted future values. Mission unchanged,
Cm utility OPEN, distinctive method and journal readiness unestablished.
