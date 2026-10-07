---
schema: ref2dex.probe.v2
probe_id: P-20261007-pointworld-ref4-input-loss-audit
experiment_id: P-20261007-pointworld-ref4-input-loss-audit
date: 2026-10-07
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: ce39282
claim_id: C3
hypothesis_family: HF-pointworld-unified-action-effect
probe_index_in_family: 2
seed_pool: probe
seeds: [216]
decision_changed_if_positive: retain PTv3 and prioritize temporal input identity and relative motion weighting for the next recipe
decision_changed_if_negative: retain the frozen current recipe and inspect other action learning explanations
status: UNCLEAR
run_id: pointworld-ref4-audit-20261007-masked-weights
---

# Ref4 action-voxel and motion-weight diagnosis

Decision: diagnose input-time identity and relative motion-weight exposure while preserving the live three-arm run.

Follow [user ref4](../../user/ref/ref4.md) without changing or stopping the live three-arm
[PointWorld Probe](P-20261007-pointworld-small-wm24.md).
This serves the Task's observed-action physical prediction goal. Results decide
which interface/recipe issue to fix before a future comparison; they do not
identify the cause of current predictive error or establish policy utility.

## Frozen protocol and resource boundary

Use the first512 entries of the existing4096-window train-only balanced draw,
seed216; verify its full index hash matches frozen normalization. Reuse the same
corpus, current cap/masks and actual microbatch2 input path. Open no validation
or test windows. Capture the unchanged model's actual pre-PTv3 unique keys and
actual loss selector with process-local mocks; PTv3 itself is replaced by a
recording seam. No parameter fitting, checkpoint inference, source/config/stat
changes, training restart, extra seed or GPU-budget extension.

CPU2threads/300seconds/<=16MiB outputs; CPU is appropriate for voxel/label
statistics. The tiny projection/embedding seam is used only to capture the
adapter inputs, with no learned PTv3 execution; GPU startup offers little
benefit for the file/statistics workload. Model, data, audit source, normalization,
manifest and train-index SHA256 are recorded and rechecked; actual runtime
commit is in result.json. Outputs:
`outputs/cm-pointflow-effect-pretrain/pointworld-ref4-audit-20261007-masked-weights/`.
Stop on input drift, nonfinite values, budget exhaustion or output conflict.

Report valid action points, distinct action voxels, time-separated voxel count,
scene/action overlap, action points sharing cross-time voxels, and same
(hand,keypoint) timestep collision/reduction. Stratify moving/static,
single/bimanual, wrist+MCP(6)/fingertip(5), both hands and all24time steps.

Report raw anchor point weight p10/p50/p90, per-window mean weights and
within-microbatch normalized weight shares relative to uniform supervision.
Stratify moving-anchor, current program, moving+program and rotation-dominant
windows. Rotation-dominant means angle>0.02rad and RMS centered-point rotational
motion>=2×RMS centroid motion. Also report angle>0.02rad with origin translation
<=2mm; define these before measurement. Surface-point displacement includes
rotation, not only rigid translation.

## Engineering reproduction

`tools/audit/diagnose_pointworld_ref4.py --repro --assert-no-time-merging`
is red-capable and has already failed on the actual adapter seam:24action
timestamps+1scene point in one1cm voxel become1backbone row. Permuting full9D
action records while reassigning timestamps leaves the pooled feature equal
within floating-point rounding. This isolates lost time/point association;
it is not a physically reversed trajectory, whose velocity features can differ.
The minimized case uses only two valid timestamps and one scene point in a
separate voxel: two action timestamps still collapse to one action row. The
remaining22timestamps are required tensor shape only and masked out.

Actual loss reproduction of40mm uniform translation over24frames gives raw
selector mean0.03445. The normalized slow-only point loss equals its unweighted
counterpart: raw3.4% does not imply a3.4%total-gradient multiplier. Relative
competition between samples/frames remains to be measured in real microbatches.
No existing hard geometry contract is reclassified as invalid by this diagnosis.

## Audit implementation correction

The initial512-window run at `f4f612e` is preserved in
`outputs/cm-pointflow-effect-pretrain/pointworld-ref4-audit-20261007/`.
Independent valid-window share conservation failed: the sigmoid capture seam
precedes the real loss's object-valid mask, and the audit omitted that later
mask in its denominator. The initial normalized microbatch shares are
INVALID_IMPLEMENTATION; voxel counts and raw anchor weights are unaffected.
The original trainer applies the mask correctly and is unchanged. Add that
mask to the audit and require valid-window normalized shares to sum to one for
every real microbatch. Repeat the affected diagnostic under the same overall
300second budget, same512indices and seed; no training or model rerun.

## Limitations / future evidence

A train-only512-window sample estimates representation/weight exposure, not
causal attribution of validation error. Overlapping windows and exposure to
the existing normalization sample prevent independent generalization claims.
Purely adding a time field to unique keys is not an implementation prescription:
PTv3/spconv is spatial3D, duplicate sparse coordinates remain a constraint, and
treating time as separate batch would suppress cross-time attention. A future
temporal-preserving adapter needs its own design/contract test. No such change
is authorized as part of this diagnostic-only request.
