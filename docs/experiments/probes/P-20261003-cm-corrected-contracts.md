# P-20261003-cm-corrected-contracts

**Status:** `PLANNED` corrected Probe; no scientific result yet.

This card records the implementation repaired after the 2026-10-03 audit. The corrected route
uses `ref2dex.cm_decision_interface.v2`, `ref2dex.cm_residual_source.v2`, and
`ref2dex.cm_residual_probe.v2`; old v1 records are frozen and rejected by the audit scripts.

The decision-interface panel contains Cup, direct-Q, Cm short-rollout, and uniform-random arms.
The previous score-permutation `shuffled` arm is removed because permuting a score vector and its
candidate indices before `argmax` leaves the selected candidate unchanged.

Residual consequence scoring rotates the predicted object-local displacement through the observed
object quaternion before extracting world-frame `z`. Candidate fitting and online selection use the
same world-height utility. Each native window snapshots motion ID, start frame, and rest height at
the trigger; reset-time metadata is never used to label an earlier window.

Each collector requires or extracts one explicit DExplore `--seed`, records it as `simulator_seed`,
and records `PYTHONHASHSEED` separately. The source fitter refuses old v1 source rows. The missing
`src.task.Cm.dataset` and `src.task.Cm.tools.data` have been restored into this checkout from the
canonical read-only source tree and pass import smoke; the corrected source and policy Probe still
need to be run.

**Stop condition:** do not train PPO or make a Cm utility claim until a corrected Probe shows valid
action ranking, sufficient non-baseline action coverage, and positive held-out native local utility.
