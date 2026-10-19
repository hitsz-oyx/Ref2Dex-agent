---
schema: ref2dex.probe.v2
probe_id: P-20261019-wam-v5-batch-audit
experiment_id: P-20261019-wam-v5-batch-audit
date: 2026-10-19
task: ptv3-shared-wam-v5
branch: main
git_commit: pending
claim_id: C1
hypothesis_family: HF-v5-batch-composition
probe_index_in_family: 1
seed_pool: probe
seeds: [177]
decision_changed_if_positive: define a new fixed-serialization computation contract before further training or executor porting
decision_changed_if_negative: retain frozen training and address loader resume and checkpoint-selection engineering separately
status: RUNNING
run_id: batch-composition-20261019-r1
---

# Frozen WAM-v5 batch-composition and training-contract audit

Classification: Decision. The user supplied a code audit identifying global
adaptive Hilbert depth, post-NCCL fork, total-objective-only best selection,
and CLI batch overrides absent from checkpoint config comparison. Check the
actual6334e1ea768 run, not unrelated later repository commits. Cheapest
discriminator: read-only inference with one published checkpoint; no optimizer
updates or hot model/statistics changes. The remote training continues.

Use a bounded panel:16 DEV candidates/source (source8 has only TRAIN), one
minimum-depth target/source,119 inherited-mixture TRAIN fillers, and the
largest-depth real neighbour found in that scan. Nine target inputs stay
unchanged across individual, batch8, and batch128 inference. This stresses
depth differences and does not estimate corpus accuracy. Seed177 is reused
exploratory diagnostic sampling, not a new independent seed.

Controls: repeated singleton; a real high-depth neighbour; a synthetic
neighbour whose observed hand alone is translated3m; a process-local fixed
depth control using the same weights. Eval disables order shuffling. Record
per-stage depth/first-sample patch membership, tau differences in mm across
K24 and first4 steps, C probability deltas, part flips and target EPE.
Synthetic and fixed-depth controls are not production input/model changes.

Frozen inputs live under remote
`/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v5-20261019-r1/`:
`outputs/finalize-training-20261019-r1/recipe/` and
`outputs/ptv3-shared-wam-v5/pretrain-nine-v5-width768-20261019-r1/`.
Copy one published checkpoint to an independent audit directory to protect
it from training rotation. Local outputs:
`outputs/ptv3-shared-wam-v5/batch-composition-20261019-r1/`.
Bound total audit output4GiB, wall time1800s, free disk>=20GiB; local GPU0
only, preserving at least two idle GPUs. Remote export is CPU-only. Do not
write in the frozen training/data/code trees.

Result: pending real-checkpoint inference; code and runtime contract review in progress.

Decision: measure impact before proposing a production serialization change.
