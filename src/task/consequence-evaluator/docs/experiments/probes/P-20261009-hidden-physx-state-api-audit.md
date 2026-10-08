---
schema: ref2dex.probe.v2
probe_id: P-20261009-hidden-physx-state-api-audit
experiment_id: P-20261009-hidden-physx-state-api-audit
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: 28e1518
claim_id: C3
hypothesis_family: HF-native-gpu-execution-contract
probe_index_in_family: 3
seed_pool: debug
seeds: []
decision_changed_if_positive: implement a native GPU hidden solver/contact-state fork and resume strict Gate1 preparation
decision_changed_if_negative: keep strict Gate1 blocked and require a new verifiable execution contract or an explicit claim decision
status: UNPROMISING
run_id: hidden-physx-api-audit-20261009-r1
---

# Does the public Isaac Gym API expose a hidden PhysX fork/restore contract?

## Decision Note

The native GPU behavior probes show contact/history divergence while visible
root, DOF, rigid-body, and RNG traces can remain close. The remaining strict
Gate1 route would require cloning or restoring PhysX contact-manifold,
warm-start, or solver-island state. This read-only audit checks the actual
archived Isaac Gym Python surface and the task integration for such an API. It
creates no simulator and uses no GPU workload.

## Audit

The archived Torch2.0.1/Isaac Gym runtime was imported without creating a
simulation. Filtered `gymapi.Gym` methods expose visible state readers and
setters: actor root/DOF/rigid-body tensors, net contact-force and force-sensor
tensors, rigid-contact queries, and sim rigid-body state getters/setters. The
filtered surface contains no `serialize`, `snapshot`, `save/load state`,
contact-manifold, warm-start, solver-island, or hidden-cache method. The raw
method inventory is in
`outputs/consequence-evaluator/hidden-physx-api-audit-20261009-r1/audit.json`.

The task-side twin contract independently fixes
`SOLVER_CONTRACT=fresh_simulator_prefix_replay` and rejects warm PhysX restore;
`REQUIRED_NATIVE_STATE_KEYS` therefore remains a visible/task-state audit, not
a claim that PhysX internals are serialized.

## Decision

The public API audit is negative for the only remaining strict same-state
implementation route. Public tensors plus Python/NumPy/Torch RNG restoration
must not be promoted to a hidden-state twin. No more synchronous-group,
serial-noise, CUDA-synchronisation, or deployment-fit simulation is justified
under the current contract. Strict Gate1 remains blocked pending a new
verifiable execution contract or an explicit decision to change the claim.
