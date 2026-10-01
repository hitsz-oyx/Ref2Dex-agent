# Randomized risk trial: first-launch engineering failure

`P-20261001-randomized-effect-risk-r1`, source0b1d2da, FAILED after35.85s in the
first panel before any acquisition window was saved or any frozen model scored.
The live player names its device `cuda`, while physical tensors use `cuda:0`.
The strict geometry bridge rejects that unresolved-device mismatch.

Fix only the bridge constructor to use the live DOF tensor's explicit device.
Do not change the geometry sampler, assignments, seed list, gates, sample size,
models or budgets. The failed run/log/manifest remain in place. Newr2is an
engineering correction of the same design; r1cost counts toward this experiment's
3600sbudget. No scientific outcome was used to select this correction.
