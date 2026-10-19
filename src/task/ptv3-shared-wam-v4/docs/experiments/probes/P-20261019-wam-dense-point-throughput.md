---
schema: ref2dex.probe.v2
probe_id: P-20261019-wam-dense-point-throughput
experiment_id: P-20261019-wam-dense-point-throughput
date: 2026-10-19
task: ptv3-shared-wam-v4
branch: main
git_commit: e0f0db7b
claim_id: C1
hypothesis_family: HF-wam-dense-geometry-cost
probe_index_in_family: 1
seed_pool: probe
seeds: [176]
decision_changed_if_positive: prepare richer surface-contact data if dense geometry has manageable training cost
decision_changed_if_negative: retain compact shared hand tokens and avoid full vertex trajectory decoding
status: PROMISING
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

## Result: dense geometry/contact is affordable; full vertex trajectories are expensive

Result: at fixed batch128, dense geometry at5mm costs13.1% more per GPU update;
also predicting2048 C queries costs20.8% more than the current10mm baseline.
Putting every hand vertex through the shared trunk/trajectory decoder costs
4.11--4.18x at matched batch64. This supports preparing richer geometry/contact
while keeping compact motion readback; it does not establish contact accuracy
or policy benefit.

Decision: prioritize778-point observed hand geometry and2048 object surface
points, with richer offline contact supervision; retain compact shared hand
tokens/trajectory outputs initially. Treat5mm as a candidate encoder grid,
not an accuracy optimum. Do not replace the production model from this timing
prototype or infer end-to-end training duration from GPU timings alone.

### Execution and verification

Four actual TRAIN GRAB windows were exported from the frozen remote reader.
Their MANO meshes use the original subject templates and H4/K24 pose samples.
The largest reconstructed11-point discrepancy is1.1921e-7m. Each2048-point
object set is distinct and on the original CAD, retaining the existing512
points. CPU geometric preparation took3.99s locally, Python3.8.20 and
PyTorch2.4.1+cu121. No GPU training was used for that preprocessing.

GPU timing: remote NVIDIA RTX4090 with49,140MiB reported memory, Python3.8.20,
PyTorch2.0.1+cu118/CUDA11.8,1024-width trunk, bf16, TF32, activation
checkpointing and fused AdamW. Each successful case warms3 and measures8
training updates. Single-hand operation is explicit. The original model and
the timing prototype give exactly equal floating outputs on the real baseline
input/checkpoint (max absolute difference0 for all seven compared outputs).
The original production model, readers, objectives and main-job config were
not edited.

| Input / output mode | Grid | Batch | Forward + loss (ms, mean) | GPU update wall (s) | Samples/s | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current512 object /11 hand, C128 | 10mm | 128 | 211.38 | .78990 | 162.05 | 7.30 |
| Current512 object /11 hand, C128 | 5mm | 128 | 215.34 | .80414 | 159.18 | 8.31 |
| Dense2048 object /778 hand; compact tau; C128 | 10mm | 128 | 222.66 | .82716 | 154.75 | 9.16 |
| Dense2048 object /778 hand; compact tau; C128 | 5mm | 128 | 244.44 | .89309 | 143.32 | 12.90 |
| Dense2048 object /778 hand; compact tau; C2048 | 5mm | 128 | 269.20 | .95416 | 134.15 | 16.64 |
| Current512 object /11 hand, C128 | 5mm | 64 | 126.97 | .45175 | 141.67 | 5.17 |
| Dense2048 object /778 shared hand + vertex tau; C128 | 5mm | 64 | 501.49 | 1.85653 | 34.47 | 17.95 |
| Dense2048 object /778 shared hand + vertex tau; C2048 | 5mm | 64 | 513.78 | 1.88628 | 33.93 | 18.30 |

Complete forward inference at batch1 is31.51ms for the current10mm model,
35.02ms for dense geometry/compact tau/C128 at5mm, and34.90ms with C2048.
The latter small differences are not evidence that more queries improve
latency. Full vertex-trajectory inference is about39.2ms at batch1. At batch128
its complete forward is969--992ms versus209ms for the compact5mm baseline.

The batch128 full-vertex cases completed inference but OOMed in subsequent
training, not inference. The main job retained about10.5GiB/card while paused,
and the benchmark reserved a2GiB margin, leaving a per-process allowance of
about34.55GiB. The OOM requested an additional4.34GiB. This does not establish
that batch128 cannot fit on an otherwise empty49GB card; the batch64 runs
establish a feasible training point within the measured resource boundary.

Dense-geometry mode preserves256 shared scene tokens and22 sparse hand slots,
plus32 goal tokens (310 total); dense vertices participate in PTv3 voxel
aggregation and sparse hand-query readback. Full single-hand vertex mode
instead uses256+778+32=1066 shared tokens and predicts24 frames for all778
vertices. The large observed cost is therefore associated with expanding the
shared/temporal computation, not KNN. No KNN/cdist was introduced.

### Recovery and timing provenance

The main1024 run had already been restored independently before this probe:
`pretrain-nine-v4-scale1024-resume16k-20261019-r1`, supervisor2975536,
torchrun PGID2975737, with its existing500GB artifact guard. It was paused
in memory under the user's explicit authorization; the script verifies PID
ownership and GPU ownership, waits for queued GPU work to quiesce, and uses
both a finally restoration path and an independent resume watchdog.

- `timing-r1`, code40f5ea29: primary matrix,149.43s pause; both full-vertex
  batch128 runs are reported as training OOM, with completed inference.
- `timing-r2`, codeeb488dc1: full-vertex batch64 and matched baseline,79.73s
  pause. Batch inference was not repeated; training was measured directly.
- `timing-r3`, codee0f0db7b: a file-not-found launch failure before any model
  execution; main restoration completed after2.09s. No timing result used.
- `timing-r4`, codee0f0db7b: separately deployed dense-geometry/C2048 case,
  completed with29.06s pause.

Cumulative pause260.31s (4min20s), below the authorized approximate10min.
GPU monitoring records the other three cards idle during each measurement
and GPU0 reaching100% in training. Benchmark updates wrote no checkpoints.
The main job was resumed after every round and restoration/progress evidence
is saved in `main-restored.json` alongside `run-manifest.json`.
The final check saw the same supervisor/torchrun PIDs, progress17480->17560,
and all four GPUs at98--99% with about10.5GiB each. Focused source syntax,
fixture shapes, experiment schema/seed ledger/index/links and cumulative-pause
checks pass; local retained probe artifacts total about2.9MB.

Limitations: four repeated GRAB windows, one active hand,7 C channels, and
GPU-resident tensors. No nine-source disk/loader/H2D pipeline, DDP collective,
validation or checkpoint overhead is included. Additional C queries are
masked supervision, so this is not a full contact-label conversion. Two-hand
full vertex decoding,16-part C output and training quality are untested.
The frozen normalization is used only to reproduce the training workload;
it is not qualified as new778-point training statistics.
Production's fixed512-point admission/checkpoint contracts remain intact;
integrating dense inputs later requires a versioned model/data contract.
