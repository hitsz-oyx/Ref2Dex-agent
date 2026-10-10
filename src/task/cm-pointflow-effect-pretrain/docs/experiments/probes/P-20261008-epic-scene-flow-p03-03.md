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

Training qualification update (2026-10-10): the geometry candidate remains
historical evidence, but its reported full hand window used side-only validity.
[The corrected readiness probe](P-20261010-video-data-readiness.md) applies all
semantic joints, verified/high-confidence quality, metadata clocks, source
splits and persistent LK masking. The first plate window now has17/28 qualified
right-hand frames; none of95 audited local candidate starts qualifies. Do not
use the historical one-window count as training readiness.

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

## Multi-clip follow-up

The available P01_03, P03_03 and P03_13 shards contain 34 clips, and all 34
metadata rows say `c2w`. Full-clip depth reprojection classifies 33 as `w2c`;
32 have at least a 5x separation and two are conservatively ambiguous
(`P01_03_3` at 1.51x and `P03_03_20` at 3.87x). The one empirical `c2w` result
is the low-motion `P01_03_3`, so it is not evidence that the metadata is
correct for the other clips.

The second complete Contact/ObjectForesight pair, `P03_13_12`, converts to
8,192 static points and 86 moved-object points with 5.8 mm/14.0 mm static
reprojection median/p95. Its Contact rows are spaced about six source frames
apart, leaving only 1 right-hand target frame valid and no H4+K24 window. This
separates the remaining blocker from the camera issue: the scene branch is
usable, while the Contact hand clock is too sparse for the current 30 Hz
window contract.

## Decision and limits

The scene-flow route is `PROMISING` for further conversion: this clip yields a
complete weakly supervised window without rigid object-pose GT. It is not yet
a frozen training dataset. The evidence is one clip, the scene labels are
depth/RGB pseudo-labels, hand joints are Contact estimates with 13 interpolated
target frames, and the release metadata contains a conflicting `c2w` label that
must be audited per clip/release. The next evidence step is a bounded multi-clip
conversion and manifest audit; training remains out of scope for this Probe.

## 2026-10-10 integration and worktree cleanup Decision Note

The user requested merging the dataset work into
`cm-pointflow-effect-pretrain` and removing the unused EPIC worktree. Merge
`a039077b2f5ec4c4349b0708353c6755db19bfde` preserves the full EPIC history
through `1f91044`; relevant converter/contact/Gate1 tests pass (14 tests),
and repository verification passes. The EPIC worktree has no tracked or
untracked user changes, but contains about 25 MiB of ignored experiment
outputs. Preserve all output files with byte hashes under the current
worktree's same relative `outputs/` paths before removing its Git registration.
Keep the EPIC branch and install an old-path compatibility symlink for
historical absolute references. The original main worktree remains because
it holds shared source datasets and ongoing independent work.

Cost: metadata, hashing, and same-filesystem moves; no GPU, model training,
large download, or checkpoint rewrite. Stop on destination collisions, user
changes, active tasks, or hash disagreement. The user has authorized removal
of the unused worktree; no additional external permission is needed. The
move manifest is local at `tmp/epic-worktree-merge-20261010/manifest.json`.

Independent implementation review found the audit's `source_clip` was
hardcoded to `P03_03_23`, including the P03_13 run. Retain original audits;
their input SHA256 values identify the sources, but the old clip field must
not define training splits or deduplication. A following provenance repair
will record the actual scene directory and explicit source start frame.
No existing scene tensors are reclassified as training data.

### Post-merge provenance repair and qualification review

The converter now derives `source_clip` from the actual scene basename and
records `source_scene_dir`, `source_start_frame`, and the actual every-other-
frame `sampled_fps=29.97002997`. Original audits and tensors are unchanged.
Engineering replay is preserved at
`outputs/cm-pointflow-effect-pretrain/epic-scene-flow-provenance-20261010-r1/`.
Both real inputs reproduce all 11 NPZ arrays exactly and have matching source
SHA256 values; the P03_13 clip field is correctly `P03_13_12`. The comparison
manifest records the repaired converter SHA256. P03_13 uses 1927 to reproduce
the historical tensor only; its metadata start1929/pad5 clock remains unresolved.
This is a CPU file/geometry replay, with no neural model computation.

The historical `PROMISING` result applies to generating candidate scene
tracks, not to qualified PointWorld training windows. Independent review,
checked against source NPZ/quality CSV by root, identifies critical limitations:

- Hand side-valid ignores `joints_valid_*` and quality/confidence. Plate has37
  invalid semantic11 entries and only20/30 fully valid source rows; bottle has29
  invalid entries and15/30 fully valid rows. The reported plate1 window does
  not establish a complete per-joint-quality training window.
- Both objects are ObjectForesight train, but the bottle's Contact labels are
  test. Preserve both source splits; do not promote it into hand training.
- LK invalid points can become valid again without verified physical-point
  identity; background points are depth-reprojected pseudo observations, not
  exact static world points. Camera reprojection self-consistency does not
  prove physical flow accuracy.
- Scene flow avoids FoundationPose/TRELLIS motion targets, but hand geometry
  still uses Contact object transforms/shape and future depth-mask fitting.
  Future hand and object labels share reconstruction sources.
- The existing PointWorld object-SE(3) loader/loss cannot consume this scene
  mask schema, and the exact source clock differs from nominal30Hz.

Keep all runs `CANDIDATE_ONLY / training_allowed=false`. Resolve source
identity/splits, per-joint validity, clock, and persistent track identity before
any matched weak-data training Probe. Detailed primary-source and current
implementation review is in
[the Task research note](../../research/2026-10-10-video-tactile-primary-sources.md).
