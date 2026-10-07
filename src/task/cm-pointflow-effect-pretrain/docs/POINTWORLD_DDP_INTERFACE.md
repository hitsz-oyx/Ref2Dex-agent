# Distributed temporal PointWorld training

User requests DDP implementation after corrected single-GPU three-arm training
starts. The three live workers remain on frozen source commit26c669c. This DDP
entry and its helpers are separate files, do not alter those sources, and do not
migrate or interrupt the active workers.

## Design and objective

Use one torchrun process per GPU, NCCL and PyTorch DistributedDataParallel.
The reference objective averages eight independently normalized TWO-sample
microbatch losses per optimizer update. Different valid-object masks and motion
weights give different denominators in each pair. Replacing them with one loss
normalized over the global batch changes the reference objective.

Keep microbatch2 and global accumulation8/effective batch16. Divide complete
reference pairs among ranks: global micro index `local_micro*world+rank`.
Each rank accumulates8/world pairs, divides losses by that local count, and DDP
averages gradients across ranks. The result is the same objective as the original
mean over eight pairs. This also preserves each pair's batch-dependent spatial
voxel origin and shuffled-donor seed. The fixed seed218draw is unchanged, with
no sampler padding, duplicate records or omitted windows. World sizes1,2,4 divide
accumulation8; unsupported counts are rejected. Campaign cap is4GPU ranks.

Use no_sync for all but the last local microbatch, including its forward pass.
Physical loss remains FP32 outside BF16 forward autocast. Clip the synchronized
gradient before AdamW. Buffer broadcast is disabled because model normalization
buffers are frozen and all ranks verify common stats/source/config/initial hashes.
History uses unused-parameter detection for its bypassed action modules.

Rank0 runs validation on the unwrapped model while peers wait for a score
broadcast, and writes logs/checkpoints. Rank0 evaluation must not call DDP forward
while peers are idle. Final weights are checked identical across ranks. All ranks
collectively report nonfinite loss and user/deadline stop conditions.

## Checkpoints and resume

Native DDP checkpoint stores unwrapped model/optimizer, global update/draw
position, config and source identities, plus each rank's torch/CUDA/NumPy/Python
RNG. Native resume requires the same world size and sources. The schedule slices
from `step*global_effective_batch`, so processed optimizer updates are not replayed.

`--import-single-checkpoint` explicitly imports a current single-GPU temporal
checkpoint with identical config/data/stats/reference source identities. This
preserves weights/optimizer/update/draw position. Rank0 restores its saved stream;
new ranks receive declared deterministic streams. World-size transition is
therefore not claimed as exact stochastic continuation. The active three-arm run
is not migrated by this implementation; import is a future explicit action.

Independent frozen temporal evaluation can read native DDP checkpoints, whose
model keys retain the original names. It additionally checks DDP source hashes
through the checkpoint's implementation-source identity.

## Invocation

```bash
TMPDIR="$PWD/tmp" CUDA_VISIBLE_DEVICES=0,1 \
/home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=2 \
  src/task/cm-pointflow-effect-pretrain/tools/run/train_oakink2_pointworld_ddp.py \
  --data outputs/cm-pointflow-effect-pretrain/oakink2-wm30-k24-20261006 \
  --stats outputs/cm-pointflow-effect-pretrain/pointworld-temporal-wm24-20261007/norm_stats.json \
  --config src/task/cm-pointflow-effect-pretrain/configs/pointworld_temporal_wm24.json \
  --output outputs/cm-pointflow-effect-pretrain/YOUR_FRESH_RUN \
  --arm action
```

This is an example future full run; assign authorized devices and shared absolute
`--deadline`, and first check resource conflicts. No new full DDP training starts
here. Three arms are three separate models, not three ranks of one model. Native
resume uses `--resume path/final.pt`; single-GPU import uses
`--import-single-checkpoint path/final.pt`. These flags are mutually exclusive.
SIGUSR1 to a verified rank0 worker requests collective checkpoint-preserving
stop. Rank PIDs are in input_manifest.json; do not signal an unknown process.

## Verification protocol

CPU two-rank Gloo tests: draw/pair reconstruction and resume slicing for1/2/4ranks,
masked objective/gradient equality to serial reference with no_sync, history-like
unused parameters, distinct per-rank RNG restore, collective stop, and rejection
of incompatible world/source identities. CPU is appropriate for this tiny
mathematical contract and avoids occupying the parallel agent's GPU3.

Then use a bounded two-rank NCCL real-data smoke on GPUs0/1, briefly sharing only
this session's identified live workers, with ample memory reserve. No unknown
process or GPU3 is touched; physical GPU budget is unchanged. Check rank-synchronized
weights and finite gradients, one-update checkpoint then resume to the second
update, standalone val-only evaluation, single-to-DDP world1 load, and the history
unused-parameter branch. Smoke saves only final.pt to bound duplicate checkpoints.
Engineering budget:10minutes,<=4GiB, no test split or scientific quality claim.
Stop on drift/nonfinite/resource conflict/deadline. Do not modify live single-GPU
model/trainer/config/evaluator sources. Outputs use
`outputs/cm-pointflow-effect-pretrain/pointworld-ddp-smoke-20261007/`.

## Files

- [DDP entry](../tools/run/train_oakink2_pointworld_ddp.py)
- [Scheduling/state helpers](../src/oakink_wm/distributed.py)
- [Distributed regression tests](../tests/test_pointworld_distributed.py)
- [Active corrected Probe](experiments/probes/P-20261007-pointworld-temporal-wm24.md)

The source may also be launched through a multi-node torchrun rendezvous with
explicit resources; current verification covers a single node. Multi-node speed,
scaling and matched science comparisons are deferred evidence.


## Executed checks

Implementation runtime `8dc508c`. All6CPU/Gloo regression checks passed in
13.18seconds. GPU artifacts and checkpoint assertions are in
`outputs/cm-pointflow-effect-pretrain/pointworld-ddp-smoke-20261007/verification.json`.

Two-rank NCCL action smoke stops after update1under an unchanged two-update
config, then native resume performs update2. AdamW state counters advance1to2,
weights change, CUDA RNG advances, and both ranks have identical parameter
hashes after each phase. Both rank RNG states are saved and distinct. The history
arm also completes a two-rank update, exercising its unused action parameters.
These are real prepared train windows with BF16 forward and FP32 physical losses.
The smoke uses global batch8to bound cost; the unmodified production config
still uses global batch16at every supported world size.

A world1import of the retained single-GPU temporal smoke checkpoint restores
step3with exact model and optimizer tensors. This is a serialization/import
check with no additional update, not a test of multi-rank stochastic continuation.
The independent temporal evaluator reads the native two-rank checkpoint and
opens only the12-sample validation panels. Cross-process metric differences
are small but not bitwise zero: max point EPE0.0503mm, translation0.00356mm,
rotation0.000393rad. No scientific predictive-quality conclusion is drawn.

All artifacts total about2.26GiB, within the4GiB engineering cap, and completed within
the10minute GPU-smoke budget. GPU0/1smoke processes exit afterward, leaving only
the ongoing corrected three arms. GPU3 is untouched. Active training's source
hashes remain identical to launch26c669c, and its workers continue finite updates.
No full DDP run or migration was launched. Only1/2rank GPU execution is verified;
4rank scheduling has CPU coverage, while4GPU/multi-node execution and throughput
are future evidence. Use the separate DDP entry when explicitly transitioning.
