# Direct randomized response: frozen implementation

Implements the previously committed P-20261001-direct-randomized-response card.
No new physical outcome has been inspected for this learner. Fit data are the
complete prior 3072 randomized windows; new test acquisition seeds are494/495.

Each nuisance fold scales and fits only the other acquisition seed. The direct
learner receives current physical geometry and reference commands, with the six
physically overwritten commands canonicalized. Factual controls use the same
context plus six pulse indicators. Every architecture is128/128SiLU; fixed
seeds611/612/613; updates1000or3000as specified. Nuisance models use the same
Adam weight decay1e-4as the response learners. Scales have floors1e-5for input
and1e-6for output numerical stability, applied using fit rows only.

The native collector gains an explicit IID assignment option, keeping its
historical balanced default. Previous experiment source identities remain in
Git and its old manifest; the new fit records both old native source hash and
current source hash. All old assets and data hashes are rechecked. New test
collection requires every fitted artifact and source to match its frozen hash.

Verification:13focused tests passed (randomized estimator, nuisance-centering
identity, independent assignment RNG, acquisition and actuation features).
All four new/modified scripts pass Python syntax compilation. Further auditing
will recompute fold scales, predictions and held-cohort risk using saved files.

Expected fit resource:GPU5,<=900s/100MiB. Test collection:GPU4sequential,
<=3600s/2GiB. Independent worktree only; original sources/checkpoints are read-only.
