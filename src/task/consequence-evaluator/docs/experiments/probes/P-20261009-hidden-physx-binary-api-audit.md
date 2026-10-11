---
schema: ref2dex.probe.v2
probe_id: P-20261009-hidden-physx-binary-api-audit
experiment_id: P-20261009-hidden-physx-binary-api-audit
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: cdef344
claim_id: C3
hypothesis_family: HF-native-gpu-execution-contract
probe_index_in_family: 4
seed_pool: debug
seeds: []
decision_changed_if_positive: implement a native binary hidden-state fork/restore bridge and resume strict Gate1 preparation
decision_changed_if_negative: keep strict Gate1 blocked and require a new runtime/API execution contract or an explicit claim decision
status: UNPROMISING
run_id: hidden-physx-binary-api-audit-20261009-r1
---

# Does the archived native Isaac Gym binary expose a hidden-state fork/restore bridge?

## Decision Note

The Python API audit found no hidden PhysX state method, but an undocumented
native bridge would still be a possible engineering route. This read-only
probe inventories the archived Isaac Gym binary and PhysX GPU plugin symbol
tables for scene snapshot/restore, collection serialization, actor cloning,
and solver/contact-state entrypoints. It creates no simulator, allocates no
GPU workload, and does not call undocumented symbols.

## Audit

The inspected runtime is
`/home2/wyy/isaac-gym/isaacgym/python/isaacgym/_bindings/linux-x86_64`.
`gym_38.so`, `libcarb.gym.plugin.so`, and `libPhysXGpu_64.so` were scanned with
both dynamic and regular symbol tables; SHA-256 hashes and the raw match lists
are retained in
`outputs/consequence-evaluator/hidden-physx-binary-api-audit-20261009-r1/audit.json`.

`libcarb.gym.plugin.so` does expose `PxCloneDynamic`, `PxCloneStatic`,
`PxCloneShape`, PhysX object `exportData`/`exportExtraData`, and RepX serializer
helpers. It does not expose a callable `PxSerialization` collection
create/serialize entrypoint, `PxCollectionExt` scene collector, or a
contact-manifold, warm-start, solver-island, or GPU-cache snapshot/restore
entrypoint. The GPU library exposes internal contact/solver/cache routines, but
no state-copy or restore boundary suitable for an external bridge.

The actor-clone symbols are not a hidden-state solution. NVIDIA's PhysX API
describes `PxCloneDynamic` as copying actor attributes while omitting joints,
scene/aggregate membership, sleep timer, and user data; its transform is not
bit-exact. RepX/binary object serialization describes API objects and their
references, not a live solver execution state. Serializing a scene while it is
being simulated is also documented as unsupported.

## Decision

The native binary audit is negative for the only remaining undocumented
implementation shortcut. The runtime has actor/object cloning machinery, but
no callable scene-level hidden solver/contact-state fork or restore contract.
Calling the visible tensor setters, actor clone helpers, or RepX serializers a
same-state twin would therefore be an unsupported claim. Keep Y, the physical
reference bank, TCC, and policy weights frozen; do not repeat synchronous-group,
serial-noise, CUDA-synchronisation, CPU/host, or deployment-fit simulation.
Strict Gate1 remains blocked pending a new runtime/API contract or an explicit
Decision Checkpoint that changes the claim.

