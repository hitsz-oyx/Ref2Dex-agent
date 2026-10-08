---
schema: ref2dex.probe.v2
probe_id: P-20261009-gpu-cuda-sync
experiment_id: P-20261009-gpu-cuda-sync
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: 0ca53ec
claim_id: C3
hypothesis_family: HF-native-gpu-execution-contract
probe_index_in_family: 1
seed_pool: probe
seeds: [282]
decision_changed_if_positive: retain an explicit CUDA-synchronised native GPU execution contract for a later bounded replay design
decision_changed_if_negative: close host/kernel synchronisation as an explanation and require a hidden-state fork or a different replay contract
status: UNPROMISING
run_id: gate1-gpu-sync-probe-20261009-r1-r2
---

# CUDA synchronisation as a native GPU replay probe

## Decision Note

Fresh native GPU runs with the same seed and checkpoint sometimes changed from a
full grasp to near-zero lift. This bounded engineering probe tests whether host
launch ordering is a material cause by changing only `CUDA_LAUNCH_BLOCKING=1`.
It does not relax the strict same-state Gate1 contract and does not score a
candidate or train an evaluator.

The cheapest discriminating test is two fresh single-environment 542-step
baselines, followed by one 4-environment group using the same environment
contract. A positive result requires both reduced baseline spread and a usable
zero-pair/group contract; stable behavior alone is insufficient.

## Frozen contract

- native `gpu_physx_gpu_pipeline`, GPU PhysX, GPU tensor pipeline, seed 282;
- source checkpoint `GRAB_00000260.pth`, current physical reference/TCC inputs;
- archived DExplore runtime and existing actor batch contracts;
- no code or policy/reference/Y change;
- outputs are engineering-only under `outputs/consequence-evaluator/`.

## Results

Two fresh single-environment baselines with `CUDA_LAUNCH_BLOCKING=1` both
preserved the native behavior: maximum lift `0.7948/0.7871 m`, held `483/483`
frames, and controlled final geometry. Their visible state traces still first
diverged at tick `78` (contact buffers diverged at tick `44`), so the traces
were not exact twins. The RNG hashes remained equal, while only `44/543`
full state hashes matched. The comparison summary is
`outputs/consequence-evaluator/gate1-gpu-sync-probe-20261009-r1/sync-audit.json`.

The 4-env synchronous group under the same setting did not preserve the group
behavior or noise contract: env0 reached only `0.1862 m` and held `10` frames;
the zero-pair pre-query p95 exceeded the declared field tolerances for hand,
joint position/velocity and object velocity, and the zero-pair gate failed.
The candidate/effect packet is
`outputs/consequence-evaluator/gate1-gpu-sync-group-20261009-r1/group.pkl`,
with the compact audit in `sync-group-audit.json`.

## Decision

`CUDA_LAUNCH_BLOCKING` appears to reduce the extreme full-episode behavior
spread for two single-env launches, but it does not restore the hidden contact
state or produce a valid synchronous candidate group. Treating this as a Gate1
noise floor would therefore be unsupported. The synchronisation-only route is
closed; strict Gate1 still requires a verifiable hidden-state fork/restore or a
new execution contract. No evaluator, PointWorld, Execution Bridge, MPC, Y,
reference-bank, or policy change follows from this probe.

## Limits

This is two blocking baselines and one group launch at one seed. It is an
engineering attribution probe, not a deterministic-physics or task-success
claim. The old ref13 binary panel was separately audited as unavailable and
incomplete for the current packet contract; its summary cannot substitute for
this native-GPU evidence.
