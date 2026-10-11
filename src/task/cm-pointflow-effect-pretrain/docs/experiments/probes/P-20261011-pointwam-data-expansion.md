---
schema: ref2dex.probe.v2
probe_id: P-20261011-pointwam-data-expansion
experiment_id: P-20261011-pointwam-data-expansion
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 3bae27c2629a7bb57e6de23e272e08e7fa583091
claim_id: C1
hypothesis_family: HF-pointwam-data-pipeline
probe_index_in_family: 3
seed_pool: probe
seeds: []
decision_changed_if_positive: admit a larger record-isolated TRAIN cohort to a matched current-only learner probe
decision_changed_if_negative: stop data-only expansion and revisit representation/source coverage
status: RUNNING
run_id: pointwam-data-expansion-20261011-r1
---

# PointWAM official-TRAIN data expansion

This Decision Probe tests whether the previous held-record failure is partly a
small-cohort problem. It keeps the verified PointWAM-style scene/hand contract,
teacher backends, substitute learner, and record-isolated split while expanding
task and episode coverage. It does not test the paper's unreproduced
Mosaic3D/Recap-CLIP/Volt architecture or make a PointWAM/Cm/policy claim.

The frozen input is official EgoDex TRAIN Part2. For each of its thirteen
inventory tasks, record `0` is `fit_candidate` and record `1` is
`dev_candidate`; no source episode crosses the split. Each episode asks for the
first two unchanged 1-second windows that pass the predeclared wrist-motion
criterion. The selection contains 26 episodes and has SHA256
`73263f1ae7c4882b5fd2e1aad03f6334102142ca0d1f9d6a454743a9ffa3ca4d`; the raw
paired members total 187,067,039 bytes. The same 31-position/30-target packs,
10 semantic hand points, initial-only scene identity selection, masks, language,
and source provenance are required.

The remote execution checkout is `/home/wbcd/workspace/oyx_ws/Ref2Dex-agent` on
`deepcybo-4-4090-48`, branch `cm-pointflow-effect-pretrain`, commit
`3bae27c2629a7bb57e6de23e272e08e7fa583091`. Raw inputs, processed packs,
manifests, and logs belong under
`/mnt/ugreen_nas/storage/oyx_storage/pointwam-data-expansion-20261011-r1/`.
The existing sibling native-data directories are audit-only until their schema
and provenance match this PointWAM contract; they are not silently mixed in.

## Protocol and gates

- Acquire only the paired HDF5/MP4 members listed in the frozen selection. Keep
  the official Part2 ETag, archive size, member CRCs, and file SHA256s. Never
  materialize the 327 GB archive.
- Process the episodes in fresh group output directories with the pinned
  VGGT-Omega and CoTracker3 sources. A group failure caused by malformed source,
  clock, hash, geometry, or resource state stops that group; a declared lack of
  qualifying motion windows is recorded as a shortage.
- Require all stored-pack and source-identity audits. The expansion is positive
  only if at least 40 of the 52 declared clips qualify, at least 10 distinct fit
  records and 10 distinct dev records qualify, and at least 8 tasks contribute.
- Use at most four currently free GPUs, keep raw acquisition below 1 GiB and
  processed expansion outputs below 4 GiB, and never overwrite a prior run or
  stop an unknown process. A remote disconnect pauses the run rather than
  changing the selection or output identity.

## Decision and follow-up

If the qualification gates pass, build a new combined manifest and run a
matched current-only learner using the same seed `247`, AdamW learning rate
`1e-3`, 1024 sampled scene points, and 600 updates as the last optimizer
diagnosis. The expanded learner must derive vocabulary and normalization from
fit packs only and keep dev records evaluation-only. If the gates fail, retain
the qualified artifacts for diagnosis but stop data-only expansion; do not call
the failure a PointWAM architecture result.

## Limitations

This is still a small slice of official TRAIN, not the paper's 1.15M-episode
human corpus. The palm offsets, offline teacher geometry, tracker query
partitioning, and substitute forecaster remain the previously declared
approximations. Success at this gate only establishes a usable larger data
cohort; it does not establish generalization, policy benefit, or exact paper
reproduction.
