---
schema: ref2dex.probe.v2
probe_id: P-20261024-workspace-origin-preflight
experiment_id: P-20261024-workspace-origin-preflight
date: 2026-10-24
task: ptv3-shared-wam-v6
branch: main
git_commit: recorded-in-receipt
classification: Decision
claim_id: workspace-origin-minus4-transparent
hypothesis_family: nine-source-supervision-distribution
probe_index_in_family: 2
seed_pool: probe
seeds: [170]
decision_changed_if_positive: allow a new workspace identity, then one C2-matched flow-only 500-update arm
decision_changed_if_negative: do not treat origin [-4,-4,-4] as a transparent repair and do not launch that arm
status: MEASURED
run_id: workspace-origin-preflight-20261024
---

# Workspace origin preflight, no optimizer

This card is the interpretation table for the owner-authorized continuation.
It is not a training run. Updates authorized: 0. The candidate change is
voxel origin [-2,-2,-2] to [-4,-4,-4] at the existing voxel size 0.005 m.
Original C1-C5 outputs stay untouched. C4-source1 is not rerun here.

The next training arm, only if this preflight passes, would match completed C2
on support, source weights, local3000 initialization, fresh optimizer, global
batch 384 and 500 updates, with the single change that `flow_future` is trainable.

## Result table

| Observed result | Meaning | Decision |
|---|---|---|
| Known source4 window s4-34183 tick 61 still rejected, or any of the 192000 C2 prescribed draws is out of range or non-finite | The -4 m origin is not a wide enough bound | Do not call the repair route false. Do not train. Pick a different bound later. |
| Every checked window encodes, but old-valid panel points move by more than 0.01 mm, the three panel means move by more than 0.001 mm, or a sample's voxel-group count changes | Origin -4 m changes the pretrained encoder's inputs | Reject "transparent repair". Do not compare a later run on this grid to local3000 as the same identity. Do not launch the flow-only arm on this grid. |
| Known window passes, 192000 draws have 0 out-of-range and 0 non-finite, panel mean absolute deltas are ≤ 0.001 mm, per-point prediction delta ≤ 0.01 mm, voxel-group counts match | The -4 m origin is a new but equivalent workspace identity for this initializer | Old C3/C5 failures stay failures. A separate C2-matched flow-only 500-update arm becomes allowed. |

Measured 2026-10-24, 0 updates, 249 seconds, GPU 0 then released. Receipt:
`/mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending/diagnosis/workspace-repair-preflight/result.json`.

The known source4 window passes at origin -4 m and still fails at -2 m. All 192000 C2 prescribed draws, 125718 unique windows, have 0 out-of-range points. The initializer panel does not stay equivalent: maximum point prediction change 41.351 mm, 86 of 512 rows change voxel-group count, and the three mean deltas versus local3000 are +0.0246 / +0.0569 / +0.0043 mm. Verdict: NOT_TRANSPARENT. The flow-only arm described by this card is not launched.

This negative result retires origin -4 m. It does not authorize a training run, and it does not forbid a separately registered ablation on the original origin. That ablation is `P-20261024-c2-flow-old-workspace`.

Flow-only thresholds, unused because the preflight did not pass, against C2's measured regression of +2.436 / +9.412 / +1.140 mm:

| Observed result | Meaning | Decision |
|---|---|---|
| All three deltas versus local3000 ≤ 0 and mother-cluster CI95 upper bounds < 0 | Freezing the shared trunk removed the C2 regression | Keep flow-only as the working repair |
| All three residuals ≤ half of C2 (+1.218 / +4.706 / +0.570 mm) but not all below 0 | Mechanism moved, 500 updates or the learning rate did not finish the recovery | Do not reject flow-only |
| Run completes, at most 20% of updates have pre-clip grad norm > 1, and all three residuals stay ≥ 75% of C2 (+1.827 / +7.059 / +0.855 mm) with CI95 lower bounds > 0 | Shared-trunk updates were not the main cause | Reject "freeze the shared trunk" |
| Residuals fall between 50% and 75%, or >20% of updates are clipped | Not identified | No route verdict |
