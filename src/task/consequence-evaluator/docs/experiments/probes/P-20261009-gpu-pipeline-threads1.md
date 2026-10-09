schema: ref2dex.probe.v2
probe_id: P-20261009-gpu-pipeline-threads1
experiment_id: P-20261009-gpu-pipeline-threads1
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: 98cec12
claim_id: C3
hypothesis_family: HF-native-gpu-execution-contract
probe_index_in_family: 7
seed_pool: probe
seeds: [282]
decision_changed_if_positive: retain GPU PhysX/GPU pipeline with one PhysX task thread for a follow-up replay contract Probe
decision_changed_if_negative: close thread-count scheduling as an explanation and retain only the original GPU behavior container
status: UNPROMISING
run_id: gpu-pipeline-threads1-20261009-r1
---

# GPU pipeline with one PhysX task thread

## Decision Note

The original native contract uses GPU PhysX, GPU tensor exchange, GPU actor
inference, and `physx.num_threads=8`. CPU PhysX and the GPU-PhysX/CPU-tensor
contract changed the policy distribution, while the native GPU group still
shows contact-stage divergence. This bounded engineering Probe changes only
the PhysX task-thread count to `1`; it keeps the GPU pipeline and all policy,
reference, TCC, and `Y` inputs fixed.

The decision is whether CPU-side PhysX task scheduling contributes enough to
the contact divergence to justify a follow-up. A four-environment 72-step
shared-prefix group is the cheapest test. It remains engineering-only and
does not relax strict Gate1 or train an evaluator.

## Frozen contract

- backend `gpu_physx_gpu_pipeline_threads1`;
- seed 282, current self-trained checkpoint, one repeated motion;
- four environments, 64 actor copies per environment (256 rows total);
- query tick 48, 72 total control steps;
- zero pair `[0,1]`, positive/negative candidates `[2,3]`;
- one GPU2 launch, no full-horizon scoring and no output video/checkpoint.

## Results

The launch completed in about 18 s. GPU PhysX, the GPU tensor pipeline, and
GPU actor inference were retained, with only `physx_num_threads=1`. The
72-step baseline reached `0.2740 m` and held 13 frames. The selected zero pair
had object-pose pre-query p95 `8.36e-4 m`, max `7.86e-2 m`, and the first raw
contact-force difference still appeared at tick 44; contact-force p95 was zero
while its max was about `19.5`.

The physical-bank audit returned environment-order `Y`
`[-0.011617, 0.011970, -0.016655, 0.015730]`. The zero-pair scalar median
noise was `0.023587`; positive/negative contrasts against baseline were
`-0.005038/+0.027347`, so the negative residual was selected only because the
noise floor was larger than the candidate difference. The packet remained
engineering-only and failed all strict calibration gates.

## Decision

Changing the PhysX task-thread count while retaining the native GPU tensor
pipeline does not restore the candidate contract. Close this backend variant;
do not run a 542-step confirmation. The original GPU pipeline remains the
behavior reference, while strict Gate1 still requires a verifiable hidden-state
fork or an explicitly changed estimand.

## Limits

One short launch cannot establish deterministic replay, policy utility, or a
population result. A positive short behavior signal would only permit a
separately budgeted full-horizon backend Probe.
