---
schema: ref2dex.probe.v2
probe_id: P-20261024-c2-flow-old-workspace
experiment_id: P-20261024-c2-flow-old-workspace
date: 2026-10-24
task: ptv3-shared-wam-v6
branch: main
git_commit: recorded-in-receipt
classification: Decision
claim_id: c2-regression-is-shared-trunk-update
hypothesis_family: nine-source-supervision-distribution
probe_index_in_family: 3
seed_pool: probe
seeds: [170]
decision_changed_if_positive: on C2 realized support, shared-trunk updates are the main cause of the C2 regression
decision_changed_if_negative: reject that claim; do not call flow-only a nine-source repair
status: PLANNED
run_id: c2-flow-old-workspace-20261024
---

# Old-workspace C2 flow-only ablation

This card is a new experiment. It is not the training arm named by
`P-20261024-workspace-origin-preflight`. That card measured origin -4 m as
NOT_TRANSPARENT and its flow-only arm stays unlaunched. This arm keeps voxel
origin [-2, -2, -2] and voxel size 0.005 m, the C2 config file, C2 source
weights, the local3000 initialization, a fresh optimizer, world size 4 and
global batch 384. The only training change is that `flow_future` is trainable
for 500 updates. C4-source1 is not rerun. No workspace repair is attempted.

C2 did not persist per-step window ids. `Steps` calls
`DiagnosticWindows.draw(96, 170, step, rank)`, whose RNG is
`SeedSequence([170, step, rank, 20261023])`. The training loop has no
reject/resample branch. `__getitem__` is a pure function of the index.
Matching the replayed index sequence is the realized sample path.

A win or a loss speaks only about that realized support. The known source4
window s4-34183 tick 61 remains illegal at origin -2 m and is outside this
arm. Do not describe a win as a deployable nine-source repair.

## Identity gate, 0 optimizer updates

| Observed result | Meaning | Decision |
|---|---|---|
| C2-flow recipe differs from C2 outside arm, stage, flow_only, recipe_path, sampling and git_commit, or the 192000 replayed (index, source, key, slot, tick) rows differ | The arm is not C2 with one trainability change | Do not train |
| Any unique replayed window is outside the fixed grid or non-finite at origin -2 m | The replay is not the path C2 already completed | Do not train |
| Replayed initializer means differ from `diagnosis/C2/panel-step000000/summary.json` by more than 0.001 mm on any of the three panel metrics | The initializer is not the C2 initializer | Do not train |
| Any parameter outside `flow_future` has `requires_grad`, or one eval batch differs from the unrestricted model | The freeze is not the only numerical change | Do not train |
| All four checks pass | Matched old-workspace ablation | Launch the 500-update arm |

Receipt directory:
`/mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending/diagnosis/C2-flow-identity`.
Training output, only after a pass:
`.../diagnosis/C2-flow`.

## Training result, only after the identity gate passes

Compare the finished panel with local3000. C2's measured regression is
+2.436 / +9.412 / +1.140 mm. `clip_grad_norm_` returns the pre-clip norm;
`result.json` records one value per update.

| Observed result | Meaning | Decision |
|---|---|---|
| All three deltas versus local3000 ≤ 0 and mother-cluster CI95 upper bounds < 0 | Freezing the shared trunk removed the C2 regression on this support | Keep flow-only as the working account of that regression |
| All three residuals ≤ half of C2 (+1.218 / +4.706 / +0.570 mm) but not all below 0 | Mechanism moved; 500 updates or the learning rate did not finish the recovery | Do not reject flow-only |
| Run completes, at most 20% of updates have pre-clip grad norm > 1, and all three residuals stay ≥ 75% of C2 (+1.827 / +7.059 / +0.855 mm) with CI95 lower bounds > 0 | Shared-trunk updates were not the main cause | Reject the claim |
| Residuals fall between 50% and 75%, or more than 20% of updates have pre-clip grad norm > 1 | Not identified | No route verdict |
