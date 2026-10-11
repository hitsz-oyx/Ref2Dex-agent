---
schema: ref2dex.probe.v2
probe_id: P-20261010-official-video-data-subset
experiment_id: P-20261010-official-video-data-subset
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 455b6c4
claim_id: C1
hypothesis_family: HF-official-video-data
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: audit a matched official DROID episode and consider a native adapter Probe
decision_changed_if_negative: stop official PointWorld data acquisition and retain native sources
requested_intervention: official_video_data_training
status: UNCLEAR
run_id: official-video-data-subset-20261010-r1
---

# Can a bounded official DROID video subset enter the native data route?

Class: Decision. This is the corrected open-source-first follow-up. The previous
checkpoint-transfer run used official weights rather than official video data and
is retained as `INVALID_IMPLEMENTATION / INCONCLUSIVE` for this question.

Question: can one official PointWorld-DROID processed flow shard be restored,
validated and converted into a reproducible sample set without downloading the
multi-terabyte corpus? A positive result permits a small data-adapter Probe;
failure stops acquisition and leaves the native sources unchanged. This audit
does not yet train a model or claim video benefit.

## Decision note / frozen audit

The full published DROID package is reported at roughly 3.9 TB compressed and
exceeds the Campaign's 300 GB local cap. Use the smallest documented subset:
`droid/flows-fs-optimized/shard-000000`, the shared shard manifest, cameras and
the confidence artifact. Expected compressed transfer is about 5.2 GB; no depth
package or other flow shard is downloaded in this run. ModelScope was checked in
the preceding source audit with no verified PointWorld-DROID match; the domestic
mirror probe timed out, so the pinned Hugging Face revision is used through the
configured proxy. The subset remains read-only and is not copied into Git.

Stop on revision/checksum drift, an output collision, a package larger than
20 GB, a recovery/schema mismatch, or any need for full-corpus acquisition.
Inspect disk usage after download. No GPU, training, checkpoint overwrite,
remote NAS write or new branch is part of this audit.

## Inputs and outputs

Source: `nvidia/PointWorld-DROID`, revision
`dd9aaeec94bb14e27ab6b16b6e4aa0dbcf3ef56f`, using the official recovery helper
and data-pipeline README. Output root:
`outputs/cm-pointflow-effect-pretrain/official-video-data-subset-20261010-r1/`.

The following must be recorded after acquisition: exact downloaded files and
SHA256, compressed/restored sizes, shard clip/episode inventory, confidence
overlap, whether the official converter emits finite samples, and the smallest
native adapter contract that can consume those samples. No conclusion about
video usefulness is allowed from this audit alone.

## Current status

Metadata audit shows 772 flow-package files (2.67 TB total), one 3.53 GB first
flow shard, one 1.64 GB confidence package and one 19 MB camera package. The
subset estimate is 5.19 GB compressed. No official video file has yet entered
training; the previous run's native dataset remains the only training data.
