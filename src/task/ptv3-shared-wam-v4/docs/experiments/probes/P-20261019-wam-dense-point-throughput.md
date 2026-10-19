---
schema: ref2dex.probe.v2
probe_id: P-20261019-wam-dense-point-throughput
experiment_id: P-20261019-wam-dense-point-throughput
date: 2026-10-19
task: ptv3-shared-wam-v4
branch: main
git_commit: 40f5ea29
claim_id: C1
hypothesis_family: HF-wam-dense-geometry-cost
probe_index_in_family: 1
seed_pool: probe
seeds: [176]
decision_changed_if_positive: prepare richer surface-contact data if dense geometry has manageable training cost
decision_changed_if_negative: retain compact shared hand tokens and avoid full vertex trajectory decoding
status: RUNNING
run_id: dense-point-throughput-20261019-r1
---

# WAM-v4 dense hand/object point throughput

Classification: Decision about computational feasibility, with engineering
timings only. The user explicitly requested testing778 MANO vertices per active
hand and2048 object points before deciding whether to prepare richer data.
The user also explicitly authorized pausing the identified remote1024 WAM job
for about10 minutes and restoring it with its in-memory state preserved.

Distinguish geometry-encoder cost from passing every hand vertex through the
shared transformer and trajectory head, and from expanding C queries. The
cheapest discriminator is four actual TRAIN GRAB windows, reconstructed from
their original MANO pose/subject templates and CAD, with GPU-resident matched
forward/backward/AdamW timings. No scientific quality or physical-contact
accuracy is inferred. No KNN is added to the prototype or production model.

The isolated prototype uses the existing1024-width PTv3/trunk/heads/objective.
It changes tensor cardinalities only in the audit tool. Baseline outputs must
match the original model on the same real inputs/weights before timing.
Dense object sampling retains the original512 samples and adds1536 distinct
on-triangle samples. Reconstructed MANO landmarks must agree with the cached
11-point hand in the same current-object frame within20 micrometres. The
fixture preserves H4/K24 and original flow identities. Added C queries remain
unlabelled/masked: their forward/loss workload is measured without fabricating
contact negatives or claiming an improved contact dataset.

Primary matrix: current11/512 at10mm and5mm; dense geometry778/2048 with compact
11-point shared readback at10mm and5mm; full778-point shared/trajectory decoding
at5mm with128 and2048 C queries. Keep32 object-flow/goal queries,256 shared scene
tokens,7 C channels, bf16/TF32/checkpointing/AdamW and batch128 fixed. Report
single-hand operation explicitly; optional two-hand or smaller-batch results
must be labelled separately. Warm up3 training steps and measure8; no
checkpoint is written and these throwaway updates never enter model selection.

Resource boundary: remote graspenv on one physical4090, four-card main job
temporarily paused only after preparation; a separate watchdog and normal
finally path restore it. Stop benchmark processes before resuming the main job.
Pause window <=10 minutes, remote output <=100MiB, free NAS >=20GiB, finite
loss/gradients and a GPU memory reserve. On OOM report the configuration and,
only if the pause budget permits, test a labelled smaller batch. Do not change
the main job's weights, optimizer, config, dataset or checkpoint files.

Artifacts: `outputs/ptv3-shared-wam-v4/dense-point-throughput-20261019-r1/` locally
and the same relative path under the existing remote dense-v4 artifact root.
Tools: `tools/audit/prepare_dense_point_benchmark.py` and
`tools/audit/benchmark_dense_points.py`. Timing excludes source-data download,
offline mesh/contact construction and DDP unless separately measured. A short
run cannot establish model quality, convergence or full80k wall time.
