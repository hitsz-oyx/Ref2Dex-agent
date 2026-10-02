# Reused eight-step successor/value compatibility Decision

Before any new model fitting or physics, distinguish whether actual future
physical observations contain task-value headroom that the fixed learned mean
successor composition misses. Source is completed physical-gradient-control-r1
panel618. This is reused-data diagnosis, not a new held-out Validation. Models
are the fixed603/604FIT option-model-policy models;618was never used to fit
them. No refit, recalibration, alternate model/checkpoint/horizon or threshold
scan. First inspect these future-state predictions only after this card is
committed. Former one-tick/oracle and deterministic actor recipes remain closed.

ALL768episodes included:192P0and576independent rawGaussian options, eachmotion
256. Same original physical105 labels, original layout/task/SDK features and
8tick horizon. Current152 PREdecision and actualfuture152 PREdecision+8 have
SDK80 derived only from the two preceding POST states. Both PREstates occur
before the first criterion tick. Future21plan/alive/time independently rebuilt
from public references/clock; actualfuturephysics only supplies131unknown
dimensions to the ORACLE diagnosis, never to deployment or an actor gradient.
FIT-only normalizers and predictor residual statistics unchanged.

Four fixed predictions, all conditioned on SAME actually held raw option:
directQ(current152,tanhoption12), ORACLE V(actualfuture152,tanhoption12),
Cm V(predictedphysicalfuture152,tanhoption12) and physical-action-removed
V(predictedfuture152,tanhoption12). Two compositions use identical analytic
future21. Fixed models/actual outcomes, no future model selection.

Primary Brier overALL768. ORACLE_USEFUL iff actualfutureV <=90% directQ
Brier AND paired95%bootstrap oracle-minus-directQ upper<0. MEAN_GAP_PRESENT
iff learned Cm Brier >=110% oracleBrier AND paired95%bootstrap Cm-minus-oracle
lower>0. Single1000bootstrap privateCPU3558, stratified256eachmotion, common
indices across methods. Per-motion/arm/error/logloss diagnostics cannot rescue
the mandatory gates. No metric/threshold/bootstrap/count scans.

PROMISING for new-representation Decision ONLY if both flags pass. Oracle
not useful => UNPROMISING, reject recycling this value contract and prioritize
a different physical interaction distribution. Oracle useful but no clear mean
gap => UNCLEAR, do not attribute actor failures to physical-state uncertainty.
Passing both permits a genuinely different uncertainty-preserving physical
representation design. Neither flag proves Jensen bias, an exact conditional
mean, calibrated transition uncertainty, algorithm novelty or policy utility.
Any later method still needs matched ACTUAL trained actors against strong
physical-action-removed and direct-Q controls with fresh outcome evaluation.

GPU fixed-network inference, CPU independent raw/current/futureSDK/publicplan,
full NumPy forwards <=2e-5, allBrier/bootstrap/gate reconstruction. Fixed
current/future feature statistics validated independently before reuse of
primary float32 decoded inputs; no optimizer replay because zero new updates.
Complete original mesh labels/audits and source model hashes must verify.
Protect original675inputs, source618/models/initial policy and own scripts;
no overwrite, no original/external writes, no job restarts or unknown kills.
Whole<=180s/32MiB, one freshly idle GPU; zero new trajectories/updates.
