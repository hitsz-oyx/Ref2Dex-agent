---
schema: ref2dex.probe.v2
probe_id: P-20261011-pointwam-data-expansion
experiment_id: P-20261011-pointwam-data-expansion
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: f4988ecec9dd2e7ef2266d371c2a4a2b093de8ea
claim_id: C1
hypothesis_family: HF-pointwam-data-pipeline
probe_index_in_family: 3
seed_pool: probe
seeds: []
decision_changed_if_positive: admit a larger record-isolated TRAIN cohort to a matched current-only learner probe
decision_changed_if_negative: stop data-only expansion and revisit representation/source coverage
status: UNCLEAR
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
`f4988ecec9dd2e7ef2266d371c2a4a2b093de8ea`. Raw inputs, processed packs,
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

## Qualification result

The five isolated acquisition groups completed with all 52 paired HDF5/MP4
members present and source hashes matching the pinned Part2 inventory. The
teacher source commits are VGGT-Omega `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6`
and CoTracker3 `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`; the Omega checkpoint
used for qualification is `4576706117` bytes with SHA256
`c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`, and the
CoTracker checkpoint SHA256 is
`2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`.

The cross-group re-audit produced
`/mnt/ugreen_nas/storage/oyx_storage/pointwam-data-expansion-20261011-r1/manifests/pointwam-expanded-qualification.json`.
It contains 48 qualified clips: 23 fit and 25 dev clips, covering 12 fit
records, 13 dev records and all 13 selected tasks. All four qualification gates
pass (`48/52`, `12/13`, `13/13`, `13/13`), so the larger cohort is valid for the
matched learner Probe. One failed motion-window episode is retained as a
declared source shortage; it is not treated as an implementation failure.

## Matched expanded learner result

The current-only substitute learner ran with seed `247`, AdamW learning rate
`1e-3`, 1024 sampled scene points and 600 updates on GPU0. The output is
`/mnt/ugreen_nas/storage/oyx_storage/pointwam-data-expansion-20261011-r1/learners/expanded-current-only-20261011-r1/`.
All finite-gradient, both-head-gradient and exact-checkpoint-replay checks pass;
the run finished in 21.25 seconds with 60.83 MiB peak allocation. Its Probe
result is `UNCLEAR`: fit hand loss falls from `0.9809` to `0.3814`, while fit
scene loss falls only from `8.2004` to `6.7318`, below the predeclared 50%
scene-fit gate. On held dev records, hand EPE is `77.06 mm` versus `72.09 mm`
for the static baseline and moving-scene EPE is `45.85 mm` versus `38.87 mm`;
the dev gate therefore fails. The expanded cohort establishes usable data
coverage, but this small substitute still does not demonstrate held-record
generalization.

## Decision and follow-up

The qualification gates passed and the matched learner was run with vocabulary
and normalization derived from fit packs only; dev records remained evaluation-
only. Because the learner gate is `UNCLEAR`, retain the qualified artifacts but
stop this data-only substitute route. The result points to representation or
source-contract limits rather than proving that more data cannot help; it is not
a PointWAM architecture, Cm, policy or formal generalization result.

## Limitations

This is still a small slice of official TRAIN, not the paper's 1.15M-episode
human corpus. The palm offsets, offline teacher geometry, tracker query
partitioning, and substitute forecaster remain the previously declared
approximations. Success at this gate only establishes a usable larger data
cohort; it does not establish generalization, policy benefit, or exact paper
reproduction.
