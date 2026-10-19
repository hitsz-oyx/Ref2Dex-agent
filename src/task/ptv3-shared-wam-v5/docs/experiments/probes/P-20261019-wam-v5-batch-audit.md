---
schema: ref2dex.probe.v2
probe_id: P-20261019-wam-v5-batch-audit
experiment_id: P-20261019-wam-v5-batch-audit
date: 2026-10-19
task: ptv3-shared-wam-v5
branch: main
git_commit: 6026c68c
claim_id: C1
hypothesis_family: HF-v5-batch-composition
probe_index_in_family: 1
seed_pool: probe
seeds: [177]
decision_changed_if_positive: define a new fixed-serialization computation contract before further training or executor porting
decision_changed_if_negative: retain frozen training and address loader resume and checkpoint-selection engineering separately
status: COMPLETED
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
Bound total audit output4GiB, wall time1800s, free disk>=20GiB. Prefer local
GPU0, preserving at least two idle GPUs. If checkpoint transfer is too slow,
use remote GPU1 with a14GiB allocator cap and at least8GiB additional free
headroom; a CUDA OOM aborts the audit, not the training. Remote export is CPU-only. Do not
write in the frozen training/data/code trees.

## Execution and evidence

Repository at audit start4530024c; model/geometry/vendor sources have no diff
from frozen training6334e1ea. Auditing code cbeb5d02 exports the panel;
6026c68c performs inference. The independently copied checkpoint is step2000,
1185762570 bytes. No live code, stats, cache, optimizer or model state was
modified; the long run was never paused or stopped during this audit.

The local checkpoint transfer was too slow and was terminated by its verified
owned SCP PID; its incomplete local file was removed. Actual inference ran
on remote physical GPU1, same graspenv/Python3.8.20/Torch2.0.1+cu118/4090 as
training, with14GiB cap and8GiB additional headroom checked before allocation.
All17 forwards completed in6.80s, peak allocated3.332GiB. Thus no cross-version
local Torch/3090 results are substituted for the actual checkpoint evidence.

Remote artifacts (relative to the remote v5 root):

- `outputs/batch-composition-20261019-r1-inputs/`: checkpoint/config/stats/panel/export identities.
- `outputs/batch-composition-20261019-r1-result/report.json`: complete inference/serialization trace.
- `outputs/batch-composition-20261019-r1-runtime.json`: live step2840 and DEV summaries.
- `outputs/batch-composition-20261019-r1-runtime-after.json`: subsequent live step3000, DEV in progress.

Local evidence under the declared local output root: `report.json`,
`summary.json`, `export.json`, `runtime.json`, `contract-review.json` and
`resume-contract/finding.json`. Large weights remain in the remote audit
directory, protected independently of training checkpoint rotation.

## Finding1: structural batch dependence confirmed; small early-panel deltas

The CPU encoding repro used the actual vendor Hilbert implementation:
512 fixed seed177 points, depth9→10, changed128-point patch membership512/512.
The real panel also contains depth10 neighbours. All nine targets separately
have depth9. Their batch128 has depth10; in target0's first layer all1726
voxels change Hilbert/Hilbert-trans patch membership, while z/z-trans remain
unchanged. At later levels depth changes from9/8/7/6/5 to10/9/8/7/6. Therefore
this is not just a hypothetical synthetic serialization difference.

| Comparison to unchanged singleton | Targets | Mean tau displacement, macro mm | Largest point displacement mm | Largest C confidence change |
| --- | --- | --- | --- | --- |
| Repeat singleton | 1 | 0.06418 | 0.33489 | 0.0001831 |
| Real batch8 (depth9) | 9 | 0.09401 | 2.01194 | 0.00390625 |
| Real batch128 (depth10) | 9 | 0.11758 | 1.82590 | 0.00390625 |
| Real higher-depth neighbour | 1 | 0.07824 | 0.33489 | 0.0001831 |
| Synthetic distant-hand neighbour | 1 | 0.08644 | 0.42626 | 0.0001831 |
| Process-local fixed-depth singleton vs neighbour | 1 | 0.07220 | 0.36848 | 0.0001831 |

For batch128, per-target mean displacements range0.02480–0.27857mm; first4
horizon mean displacements range0.00669–0.06821mm. Per-target change in EPE
is at most0.03206mm. These are prediction differences, not ground-truth EPE.
Only target0's serialization trace is captured in the batched runs; other
targets' output comparisons are measured independently of that trace.

Repeated singleton already varies under bf16/sparse GPU reductions. Fixed-depth
control still varies, so total deltas cannot be assigned to depth alone.
This small, deliberately selected early-checkpoint panel does not establish
batch invariance, negligible late-training impact, or a universal millimetre
bound. It does not justify declaring the current training invalid. Recheck a
later meaningful checkpoint before executor porting; do not hot-change depth.
If depth is fixed later, audit every pooling serialization stage as well as
the initial stage (GridPooling calls serialization again).

## Finding2: fork stability and STOP limits confirmed in code

`tools/run/train_pretrain.py:93` initializes NCCL, before DDP and the loader;
line126 chooses fork, with8 workers/rank. The loader has no explicit timeout,
and line167 obtains the next batch before checking STOP. The supervisor's
180s grace can terminate its owned group if that read hangs; there is no
guarantee that such a hang publishes a final checkpoint.

This matches the [official PyTorch DDP warning](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)
to use spawn/forkserver with NCCL multiworker loading. No actual deadlock was
observed; startup/updates do not certify fork safety. Before the next
launch/resume, change the loader context in a new snapshot, check pickling and
actual four-rank read/update/DEV/stop/recovery. No spawn throughput or four-GPU
resume-equivalence result is claimed from this read-only audit.

## Finding3: total-objective-only selection can discard a better tau candidate

Training line216 selects only source_macro_objective; retention line158 adds
only that best to first/milestone/recent protections. There is no trajectory
selection or tau-protected checkpoint. This is a selection risk, not a current
demonstration that best.pt has become worse at tau.

Actual eight-source fixed DEV results:

| Update | Macro objective | Hand EPE mm | Root-relative hand mm | Contact-phase hand mm |
| --- | --- | --- | --- | --- |
| 20 | 5.12878 | 59.6863 | 19.9487 | 52.9042 |
| 1000 | 2.19733 | 47.8947 | 19.6682 | 42.5220 |
| 2000 | 1.64897 | 42.8652 | 19.2715 | 37.1280 |

Tau improves alongside C; the observed loss reduction is not C-only. However,
root-relative hand EPE improves only3.4%, versus28.2% for full hand EPE.
That supports tracking articulation/root-relative and short/contact metrics,
not assuming improved hand shape from overall EPE. This is early DEV on one
seed, not deployment or v4 superiority. At2000, macro-objective best and
macro-hand-EPE best coincide; its full checkpoint is separately protected by
the audit copy. A prospective additional tau selection/retention rule remains
needed; this audit does not silently replace production best.pt semantics.

## Finding4: effective-batch resume hole reproduced through actual entrypoint

The line81 CLI batch override changes the loader but not the config compared
at line110 or saved in checkpoints. An actual CPU optimizer/checkpoint/resume
loop used the existing tiny geometry/data fixture (debug seed0): save at
update1 with configured batch1, resume with the same JSON and --batch-size2,
complete update2. Both checkpoints still say config.batch_size1, while the
resumed run-manifest/progress correctly reports actual batch2. This confirms
a real entrypoint gap, beyond a dictionary-equality illustration. The normal
four-GPU launcher passes no override, so this is not evidence of a live
batch512 violation. Bind effective local/global batch to future checkpoint
contracts and explicitly migrate older checkpoints; do not call changed-batch
continuation strict resume.

## Other report claims and limits

The shared local C readout, repaired Shadow caches and H4 admission contract
are present in the frozen path; no additional full cache rebuild is justified
by this audit. TRAIN-only K24x3 joint21 moments use128 windows/source1152
total, inherited main3 strata, and pooled valid point counts. They are not
full-corpus or exact optimizer-probability-weighted moments. DEV has eight
sources; source8/DexYCB remains TRAIN-only. Cache census/contact exposure still
cannot substitute for actual source/stratum-weighted optimizer exposure.
These boundaries are correctly identified in the supplied review.

The DDP objective uses global valid denominators and world compensation;
separate C denominators prevent query count alone from scaling its weight.
Static forward input inspection shows current/H4 hand data plus explicit
object t+24 goal conditioning, not future-hand labels. This is code review,
not a new four-rank gradient-equality or full data-lineage certification.

Result: the report's four engineering/structure concerns are confirmed; real
step2000 batch128 output displacement is0.118mm macro mean on the small panel,
with nonzero singleton numerical variation, while early DEV tau also improves.

Decision: continue unchanged768 training. Before the next launch/resume,
repair loader/effective-batch contracts and validate the full recovery path;
preserve a predeclared tau candidate alongside objective-best. Revisit batch
sensitivity at a later checkpoint before deciding on a new serialization
contract. Broader normalization/contact/DexYCB audits are separate protocols.
