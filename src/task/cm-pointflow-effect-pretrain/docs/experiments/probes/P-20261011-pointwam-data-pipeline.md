---
schema: ref2dex.probe.v2
probe_id: P-20261011-pointwam-data-pipeline
experiment_id: P-20261011-pointwam-data-pipeline
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 81410ba85f18e82d007c9dd8d04a3dcd96935f50
claim_id: C1
hypothesis_family: HF-pointwam-data-pipeline
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: permit a bounded train-split data qualification after backend and visual checks
decision_changed_if_negative: repair the metric scene and semantic hand contract before training
status: UNCLEAR
run_id: pointwam-data-pipeline-20261011-r2
---

# PointWAM human scene/hand data pipeline: small engineering pilot

Class: Blocker / engineering smoke. User explicitly authorizes a few-data
pipeline run on the existing cm-pointflow-effect-pretrain worktree. This serves
the Task's human-video representation subgoal; it does not test robot actions,
Cm policy utility, or PointWAM's success-rate claim. No training is authorized
by this pilot, and official EgoDex test samples remain training_allowed=false.

Question: Can an already-local EgoDex video/HDF5 pair be converted into aligned
metric scene trajectories, ten semantic points per hand, and language, with
explicit masks, source clocks, provenance and visual review artifacts?

The cheapest test uses at most two 1-second clips from one original episode.
Reuse existing data/environment/depth weights read-only. CoTracker3 official
offline code/weights are acquired locally. Preserve the user's untracked
pointwam_core.py; this runner uses a separate NumPy data-contract module.

## Protocol and resource boundary

- 30Hz source; 11 frames at 10Hz, .5s candidate stride; one fully observed wrist
  must have >=2cm summed displacement. Resample to31 positions,30 future targets.
- Each human palm frame has middle-knuckle origin, orthogonal wrist/knuckle-row
  axes, six fixed Allegro offsets and four actual fingertips. Each hand is
  constructed independently. Offset source/quantization/axis convention are
  explicit; unquantized author constants are not available.
- 640x360 RGB; first-frame queries every2pixels, original query IDs retained.
  CoTracker3 offline; no backward tracking; bounded query partitions include
  the official6x6 support grid. This partitioning is not bitwise author replay.
- R1 uses already-local SpaTrackerV2 VGGT4Track depth as an explicit substitute.
  R2 uses VGGT-Omega-1B-512 from the user-selected ModelScope mirror, verified
  against its published SHA256 and loaded strictly into pinned official code.
  Original HF checkpoint access remains gated; its LFS SHA cannot be compared.
- Metric scale uses observed native joints' z/depth ratios, factor-two median
  filtering; native camera transforms lift tracks into ARKit origin, then the
  scene and hands receive the same proper z-up rotation.
- Select initially visible/depth-valid identities,1.5cm voxel dedup,12000 cap.
  RGB, instruction, all observation masks and target masks remain in the pack.
  Offline teacher reconstruction is not certified causal perception.
- <=1GPU, <=60minutes processing wall time, <=6GB new artifacts including the
  tracker/Omega weights/code; Omega acquisition is capped at20minutes. No native dataset modification, no model training,
  no existing checkpoint overwrite. Stop on clock/geometry/schema mismatch.
- Engineering checks:31/30 shapes;>=256 scene points;one complete10-point hand;
  scale relative IQR<=.35;native camera projection round-trip<=.01pixel;finite
  valid points;increasing clocks. Human visual review remains a separate gate.

## Inputs and outputs

Tool: `../../../tools/run/prepare_pointwam_pilot.py` (Task-relative entry:
`tools/run/prepare_pointwam_pilot.py`). Output:
`outputs/cm-pointflow-effect-pretrain/pointwam-data-pipeline-20261011-r1/` and
`outputs/cm-pointflow-effect-pretrain/pointwam-data-pipeline-20261011-r2/`.
Machine-local external input paths, SHA256s, backend identities and actual
execution commit belong to the run manifest. Sources remain read-only.

Result: VGGT-Omega/CoTracker3 completed two clips in97.76s; numerical gates and stored-pack audit pass; visual palm qualification remains pending.
Decision: review real-data visual alignment before any training or scaling.
Unquantized palm constants and exact author preprocessing remain future
evidence, along with train-split acquisition and robot-stage transfer.

## ModelScope steering and bounded backend rerun

User supplied `https://modelscope.cn/models/facebook/VGGT-Omega`. Its public file
list and pinned download endpoint are accessible through the configured proxy;
the 512 checkpoint is4576706117bytes, SHA256
`c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`,
file revision `f4fe2040e400351980060e5391e12f00a6c037b2`.
HF public metadata confirms name and size but masks LFS SHA; do not claim a
cross-registry hash comparison. Verify bytes against the ModelScope SHA, load
strictly into the pinned official source, and retain this provenance boundary.

Decision note: the cheap substitute r1 completed both clips with numerical
engineering gates passing. The mirror enables replacing only the depth backend
with VGGT-Omega on the same clips (r2), preserving r1 as substitute evidence.
New weights fit the amended6GB local budget and global300GB limit; existing
outputs are approximately120GB in main plus37GB in this worktree. No extra
permission is required within the user-authorized pipeline scope. Stop on hash,
model-key, resource or geometric mismatch; no training or data scaling follows.

The depth-resize helper now preserves invalid bilinear support. Both r1 depth
outputs have zero missing/nonfinite pixels, so this repair does not invalidate
its observed depth fields; it closes the future input contract. R2 uses it.

## Observed engineering result

R1 ran at `6625a06` in102.76s with the declared depth substitute; both clips
passed their numerical gates. Preserve it as engineering evidence only.
R2 ran at `81410ba85f18e82d007c9dd8d04a3dcd96935f50` in97.76s, on physical
GPU7, peak PyTorch allocation6923.70MiB. Tracking utilization was100%; GPU7
returned to idle after the process exited. The two clips overlap by.5seconds,
so these are not independent samples. No model training was run.

| R2 clip | Source frame IDs | Initial scene points | Tracks valid all31frames | Scene valid fraction | Hand valid fraction | Scale relative IQR | Roundtrip max px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip00-f0000 | 0,3,...,30 | 10755 | 6162 | .78692 | 1.0 | .14750 | .000252 |
| clip01-f0015 | 15,18,...,45 | 9445 | 5355 | .73892 | 1.0 | .12737 | .000252 |

Each scene shape is `(31,N,3)` and hand shape `(31,2,10,3)`, right then left.
Each future displacement contains30steps relative to t0. Missing points remain
NaN with false masks; target masks include t0 validity. A separate stored-pack
audit verifies actual clocks, identity/RGB/language, finite/missing contracts,
proper rotation, displacement values/masks and training exclusion. All14checks
per clip pass. Seven contract tests pass, covering camera motion removal,
missing-data recovery, palm geometry, scale rejection, initial selection and
valid-support depth resizing. Pure geometry/file audits use CPU; both teacher
models run on GPU.

The download completed in186.33s with exact byte/SHA agreement. Official source
identities are VGGT-Omega `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6` and
CoTracker `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`; all input, weight and
executed source-file hashes are retained in the manifest. R2 generated57.05MB
before the supplementary stored-pack audit; total new assets remain below6GB.

- [R2 run manifest](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-data-pipeline-20261011-r2/manifest.json)
- [Stored-pack audit](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-data-pipeline-20261011-r2/stored-pack-audit.json)
- [ModelScope download verification](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-pipeline-assets-20261011/omega-download.json)
- [Clip00 RGB/depth overlay](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-data-pipeline-20261011-r2/clip00-f0000/projection_depth.jpg) and [interactive scene/hand](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-data-pipeline-20261011-r2/clip00-f0000/scene_hand.html)
- [Clip01 RGB/depth overlay](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-data-pipeline-20261011-r2/clip01-f0015/projection_depth.jpg) and [interactive scene/hand](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-data-pipeline-20261011-r2/clip01-f0015/scene_hand.html)

Root inspected both RGB/depth overlays and static3D previews. Original fingertip
labels follow the visible fingers; the six fixed palm points sometimes project
outside the human silhouette. They describe mapped robot geometry, so annotation
validity alone cannot qualify that mapping. User visual review should check the
hand frame and its relationship to the scene before accepting these labels.
No obvious global camera flip appeared in these views; roundtrip consistency
does not establish depth accuracy or contact alignment.

## Limitations / future evidence

Engineering only; status remains UNCLEAR for the data/method research question.
The hand offsets use .5mm-quantized public viewer robot geometry and a stated
human middle-knuckle frame, not an exact author table. Tracker query partitions
with support grids can change predictions relative to one joint dense call.
Joint-to-surface median scale is approximate and needs geometric qualification.
Reconstruction uses the full clip and is not certified causal perception.
HF/ModelScope LFS hash equality could not be checked. Do not interpret the tiny
camera roundtrip error, passing tests or two overlapping clips as metric GT
accuracy, paper reproduction, data benefit or robot-policy validation.
Before scaling: qualify palm/scene visuals and exact constants where available,
obtain official TRAIN sources, then design a bounded data qualification. Robot
stage and native-transfer claims remain separate future evidence.
