---
schema: ref2dex.probe.v2
probe_id: P-20261008-epic-scene-flow-p03-03
experiment_id: P-20261008-epic-scene-flow-p03-03
date: 2026-10-08
task: cm-pointflow-effect-pretrain
branch: agent/epic-scene-flow
git_commit: fcfe844
claim_id: C1
hypothesis_family: HF-epic-scene-flow
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: convert EPIC clips into a weakly supervised scene-flow source and audit more clips
decision_changed_if_negative: keep EPIC out of scene-flow expansion and retain native rigid sources only
status: PROMISING
run_id: epic-scene-flow-p03-03-20261008
---

# EPIC scene-flow conversion without object-pose supervision

This Probe tests the revised data contract: EPIC-Contact contributes hand and
hand-object geometry, while the RGB clip, depth, camera trajectory and object
mask produce persistent scene points. FoundationPose `T_c_o`, the TRELLIS mesh,
and propagated object pose are excluded from the candidate. It tests data
processing only; no PointWorld training is run.

## Conversion

The converter samples background points from frame-zero depth and validates
their reprojection in every target frame. It tracks object-mask RGB corners
with Lucas–Kanade and backprojects their current depth. Both are transformed
to the SpaTracker world frame. Contact hand joints are first expressed
relative to the Contact object mesh, then placed by a bounded similarity fit
between that mesh and the per-frame ObjectForesight object-mask depth cloud.
The fit is performed in world coordinates with a sequential initialization and
a fixed metric scale after the first frame; this avoids mixing the two releases'
different absolute camera origins.

## P03_03 evidence

The run used `P03_03_23` and the verified
`P03_03_4054_4116_right_plate.npz` pair. The candidate is at
`outputs/cm-pointflow-effect-pretrain/P-20261008-epic-scene-flow-p03-03-aligned5/`.

- 8,192 static points and 187 moved-object points pass the ≥0.8 track-validity
  threshold; 8,379 total points have mean validity 0.983, and 7,099 are valid
  in every target frame.
- Static depth reprojection residuals are 3.1 mm median and 14.3 mm p95.
- All 28 30-Hz target frames have a valid right-hand fit; the left hand is
  absent in this Contact record. The fit residual is 8.2 mm median and
  10.3 mm p95, and one complete H4+K24 window is emitted.
- The hand/object centroid trajectories move continuously after sequential
  fitting. The object displacement is 0.234 m and the hand displacement is
  0.215 m over the window; the largest hand centroid step is 23 mm.
- The raw SpaTracker matrices only pass the static reprojection check when
  interpreted as `w2c` (3.4 mm median versus 88.6 mm as `c2w`). The upstream
  `step9_spatracker.py` source records that it saves `inverse(c2w_traj)`.

A sixteen-clip audit found the same direction for 15 clips, while their
metadata rows all say `c2w`; one low-motion clip is only 1.52x separated and
is left ambiguous. This is a release-level metadata/artifact mismatch, not a
failure of the P03_03 trajectory itself. The converter now supports
`--extrinsics-convention auto`: it selects the lower static-depth reprojection
residual only when the two directions differ by at least 5x, otherwise it
refuses the clip. On the P03_03 run, auto selected `w2c`, recorded
`metadata_mismatch=true`, and reproduced the same 28/28 hand and one-window
result. The exact release-generation step that created the stale `c2w` labels
is not recoverable from the published artifacts, so the metadata is preserved
for audit rather than overwritten.

## Decision and limits

The scene-flow route is `PROMISING` for further conversion: this clip yields a
complete weakly supervised window without rigid object-pose GT. It is not yet
a frozen training dataset. The evidence is one clip, the scene labels are
depth/RGB pseudo-labels, hand joints are Contact estimates with 13 interpolated
target frames, and the release metadata contains a conflicting `c2w` label that
must be audited per clip/release. The next evidence step is a bounded multi-clip
conversion and manifest audit; training remains out of scope for this Probe.
