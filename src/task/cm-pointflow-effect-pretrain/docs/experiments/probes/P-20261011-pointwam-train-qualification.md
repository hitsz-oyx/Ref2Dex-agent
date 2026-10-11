---
schema: ref2dex.probe.v2
probe_id: P-20261011-pointwam-train-qualification
experiment_id: P-20261011-pointwam-train-qualification
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 898c8da
claim_id: C1
hypothesis_family: HF-pointwam-data-pipeline
probe_index_in_family: 2
seed_pool: probe
seeds: []
decision_changed_if_positive: retain an episode-isolated TRAIN pack for a bounded joint scene-hand learner
decision_changed_if_negative: repair the failed source or geometric contract before acquisition or learning expands
status: RUNNING
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

Inputs, frozen selection, fit/dev ownership and pinned archive metadata belong
to `configs/pointwam_train_pilot_selection.json` and the acquisition manifest.
Outputs: `outputs/cm-pointflow-effect-pretrain/pointwam-train-qualification-20261011-r1/`.

Result: pending bounded TRAIN qualification.
Decision: execute the frozen qualification before expanding data or training.

## Limitations / future evidence

Single small inventory slice; fit/dev are record-isolated within the official
TRAIN release, not subject/session-isolated and not an official benchmark split.
Labels are offline full-clip teacher reconstructions. Approximate palm offsets,
joint-to-surface scale, query partitioning and missing author preprocessing
remain declared. A passing data gate cannot establish scene depth accuracy,
learnability, heldout prediction, native representation benefit or policy utility.
