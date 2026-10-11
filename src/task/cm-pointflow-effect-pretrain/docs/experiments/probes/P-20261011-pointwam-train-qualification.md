---
schema: ref2dex.probe.v2
probe_id: P-20261011-pointwam-train-qualification
experiment_id: P-20261011-pointwam-train-qualification
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 13fd18ebb42aa379e8afba5bb320b0a2facd083d
claim_id: C1
hypothesis_family: HF-pointwam-data-pipeline
probe_index_in_family: 2
seed_pool: probe
seeds: []
decision_changed_if_positive: retain an episode-isolated TRAIN pack for a bounded joint scene-hand learner
decision_changed_if_negative: repair the failed source or geometric contract before acquisition or learning expands
status: PROMISING
run_id: pointwam-train-qualification-20261011-r1
---

# PointWAM official-TRAIN data qualification

Class: Decision. User accepted the two-clip visual smoke and instructed us to
continue. This serves the Task's human-video representation subgoal and leaves
the global action-conditioned Cm/policy utility claims unchanged.

Question: Does the same scene/hand preprocessing produce sufficiently observed,
aligned 31-frame packs across several official-TRAIN tasks and records, rather
than only one overlapping official-test pair? A positive screen permits a
bounded joint scene/hand learner; failures first trigger contract diagnosis.
The cheapest discriminating probe is six episodes, at most twelve clips, with
frozen selection and numerical gates. No model training in this qualification.

## Protocol

- Acquire only paired HDF5/MP4 members from official EgoDex Training Part2,
  using exact HTTP ranges and CRC checks; record archive ETag/size and file
  SHA256. Never materialize the roughly300GB archive. The official README
  identifies Part1..5 as TRAIN and test.zip as TEST:
  [author source](https://github.com/apple-aiml-research/ml-egodex#dataset-access-and-download).
- Freeze three manipulation tasks present in the inventory, two smallest
  numeric record IDs per task; assign the first to fit_candidate and second
  to dev_candidate before teacher processing. No record crosses partitions.
  Candidates are not permission to train: all pilot packs retain
  training_allowed=false until the learner's admission contract is explicit.
  The frozen tasks are basic_pick_place, basic_fold and fold_unfold_paper_basic;
  IDs0/1 belong to fit/dev respectively. Deformable scenes are appropriate for
  per-point trajectories; this qualification does not impose rigid-object6DoF.
- Per episode, first two 1-second windows passing the unchanged2cm complete-
  wrist criterion, with.5second stride,11frames at10Hz,31resampled positions.
  If fewer candidates exist, report that shortfall; do not replace records
  after observing outcomes. Overlapping windows count as one source episode.
- Reuse pinned VGGT-Omega/CoTracker3 and the user-accepted approximate palm
  convention; keep native cameras, same depth scaling/masks and initial-only
  scene selection. Official-test r1/r2 never enter these candidates.
- A qualified clip passes all original engineering gates, plus >=512 scene
  identities valid at all31frames and >=.5 scene observation fraction.
  Proceed gate: >=8qualified clips from >=2tasks, with >=2distinct fit and
  >=2distinct dev records. Below gate: UNCLEAR and diagnose; do not reject
  the broad PointWAM method from small-data teacher failures.
- File/statistics audits run on CPU; teacher inference runs on one free GPU.
  New raw input <=1GiB, new processed output <=1GiB; acquisition <=20min,
  all sequential teacher processing <=30min. Existing weights are reused.
  Stop on archive/input identity drift, split mismatch, output collision,
  time/disk cap, malformed clock or resource conflict. Preserve failed runs.

## Decision note / resources

The user accepts the prior numerical and visual pipeline. Official TRAIN
qualification is the cheapest next step before learning; full corpus processing
or building an unqualified learner is unnecessary. Current outputs are42GB in
this worktree plus120GB in main, within the shared300GB cap;329GB disk is free.
The additional2GiB bound fits. Stay on cm-pointflow-effect-pretrain and preserve
the user's untracked pointwam_core.py/ref files. No new branch, external write,
identity permission, or change to Mission claim is required.

Index preflight: the first90second/32MiB metadata attempt hit its cap. A logged
retry capped at360seconds/128MiB succeeded: Part2 is327274733628bytes, ETag
`"1bb5321e52a3204c791561dd30204968-4877"`; central-directory transfer26764538bytes.
The frozen six pairs total63242354uncompressed bytes. Download remains capped
at1GiB; the large archive is never staged.

Acquisition completed in80.68seconds,83520156range-transfer bytes including
the ZIP index; all twelve selected members passed CRC and have recorded SHA256.
The group re-verifies all source pairs, both pinned checkpoint hashes and both
teacher Git identities before starting GPU subprocesses. Source provenance is
stored in each pack as well as its manifest. TEST input is rejected by the
TRAIN qualification entry; missing/modified/incomplete TRAIN inputs are rejected.

Tools: `../../../tools/run/download_pointwam_train_subset.py`,
`../../../tools/run/qualify_pointwam_train.py`; the original per-episode runner
now accepts a hash-verified `--source-manifest`. Nine data contract tests pass,
including split/hash tampering and preservation of test exclusion.

Inputs, frozen selection, fit/dev ownership and pinned archive metadata belong
to `configs/pointwam_train_pilot_selection.json` and the acquisition manifest.
Outputs: `outputs/cm-pointflow-effect-pretrain/pointwam-train-qualification-20261011-r1/`.

Result: 12/12 TRAIN clips from3tasks/6records qualify; record-isolated fit/dev each contain6clips; processing605.94s and peak GPU allocation6923.70MiB.
Decision: retain this bounded subset for joint scene/hand learning; do not expand raw data or admit TEST samples.

## Observed qualification

The frozen selection completed unchanged at
`13fd18ebb42aa379e8afba5bb320b0a2facd083d`; every original engineering gate,
stored-pack check and extra observation floor passed. The four predeclared
qualification gates all pass:12clips,3tasks,3fit records and3dev records.
PROMISING applies only to the data qualification question, not to learnability
or a representation/policy benefit. No model training occurred.

| Original record | Partition | Source start frames | Scene points, two clips | All31frame-valid tracks | Hand annotation validity |
| --- | --- | --- | --- | --- | --- |
| basic_pick_place/0 | fit_candidate | 0,15 | 8704 /8585 | 4510 /2602 | 1.0 |
| basic_pick_place/1 | dev_candidate | 0,15 | 8695 /10085 | 5208 /3505 | .5: right only |
| basic_fold/0 | fit_candidate | 0,15 | 7294 /6690 | 5526 /4159 | 1.0 |
| basic_fold/1 | dev_candidate | 0,15 | 12000 /12000 | 6984 /2160 | 1.0 |
| fold_unfold_paper_basic/0 | fit_candidate | 30,120 | 6163 /5033 | 5571 /3944 | 1.0 |
| fold_unfold_paper_basic/1 | dev_candidate | 30,45 | 4715 /4788 | 3418 /3332 | 1.0 |

Scene observation fraction spans.57449–.93152; calibration relative IQR spans
.09870–.29350. Maximum camera roundtrip error is.000288px; this is consistency,
not independent depth accuracy. Each pack contains31positions/30targets with
source clocks, IDs, RGB/language and right/left hand order. Twelve stored-pack
checks per clip pass. Missing left-hand coordinates/targets in pick_place/1
are entirely NaN, with entirely false masks; they are never filled as origin
points. Stationary paper starts are filtered before teacher computation.

The final audit verifies that all twelve acquired source files remain unchanged,
all twelve pack hashes match, and fit/dev source episode sets are disjoint.
Root inspected task-diverse RGB/depth overlays and a new3D preview; no new
global flip or packing defect was apparent. Palm mapping remains the convention
the user accepted in the original visual pilot, with its stated approximation.

Processing wall time605.94s, about50.5seconds per clip including model reload,
file/hash checks and visualization. GPU7 showed100% utilization during tracking;
peak PyTorch allocation6923.70MiB and the GPU returned to idle on completion.
New processed data324.69MB; raw inputs plus preserved ZIP inventory94.03MB.
Both remain comfortably below their1GiB caps. A persistent copy of the pinned
inventory has the same SHA256 as the acquisition's original temporary input,
so subsequent downloads do not depend on a disposable cache.

- [Qualification summary](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-train-qualification-20261011-r1/qualification_summary.json)
- [Group manifest with per-record commands and per-clip checks](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-train-qualification-20261011-r1/manifest.json)
- [CRC/SHA acquisition manifest](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-train-inputs-20261011-r1/download_manifest.json)
- [Preserved pinned ZIP inventory](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-train-inputs-20261011-r1/zip_inventory.json)
- [Fold RGB/depth](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-train-qualification-20261011-r1/episode03-basic_fold-1/clip01-f0015/projection_depth.jpg) and [interactive scene/hand](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-train-qualification-20261011-r1/episode03-basic_fold-1/clip01-f0015/scene_hand.html)
- [Paper RGB/depth](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-train-qualification-20261011-r1/episode05-fold_unfold_paper_basic-1/clip01-f0045/projection_depth.jpg) and [interactive scene/hand](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-train-qualification-20261011-r1/episode05-fold_unfold_paper_basic-1/clip01-f0045/scene_hand.html)

Next learner admission must explicitly use these qualified official-TRAIN
sources/record partitions, current scene/hand/instruction inputs, masked joint
scene/hand targets and appropriate persistence controls. Frozen native training
and its checkpoints are separate; these packs are not a direct H4/K24/11-hand-
point replacement. Before much larger acquisition, reduce repeated teacher
model loading and reassess measured throughput/resource cost.

## Limitations / future evidence

Single small inventory slice; fit/dev are record-isolated within the official
TRAIN release, not subject/session-isolated and not an official benchmark split.
Labels are offline full-clip teacher reconstructions. Approximate palm offsets,
joint-to-surface scale, query partitioning and missing author preprocessing
remain declared. A passing data gate cannot establish scene depth accuracy,
learnability, heldout prediction, native representation benefit or policy utility.
