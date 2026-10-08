---
schema: ref2dex.probe.v2
probe_id: P-20261007-epic-contact-overlap
experiment_id: P-20261007-epic-contact-overlap
date: 2026-10-07
task: cm-pointflow-effect-pretrain
branch: consequence-evaluator
git_commit: 7c85d5df67af2f4e814c62e2b8d2c080a4a64999
claim_id: C1
hypothesis_family: HF-epic-contact-overlap
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: build a bounded same-clip EPIC hand+object converter and then test its windows as a separate mixed-source data input
decision_changed_if_negative: stop the EPIC-Contact/ObjectForesight bridge and keep expansion on native 3D sources; do not pair across datasets
status: UNCLEAR
run_id: epic-contact-overlap-20261007
---

# Same-clip EPIC-Contact and ObjectForesight bridge

This Probe asks whether the two releases can provide the two halves of one
training window without cross-dataset temporal pairing: EPIC-Contact supplies
the posed hand trajectory and ObjectForesight-EPIC supplies the object mesh and
6-DoF trajectory for the same EPIC source clip. It serves the Task subgoal of
expanding the current 30 Hz, history-4, horizon-24 `H+A+Z` corpus. It does not
test PointWorld learning, policy utility, or Cm utility.

## Cheapest discriminating test

First download only the public quality CSVs and ObjectForesight metadata. Use
the shared `video_id` and source frame numbers to identify candidate clip
overlap. Then inspect one small ObjectForesight shard (target `P03_13`, whose
shard is about 146 MB) and try to form at least one continuous 4+24 window.
The audit must reject non-contiguous frame IDs, pose reinitialization changes,
ambiguous camera direction, mesh/pose scale mismatch, and windows that do not
share the annotated EPIC-Contact clip interval. EPIC-Contact's own hand and
object fields are converted first; ObjectForesight is used only as an
independent same-clip/frame audit source and is never mixed into this candidate
pack without a separate scale and camera-contract decision.

## Frozen contract and quality tiers

- Pair only records from the same EPIC source video and overlapping source
  frame numbers. Never attach an EgoDex or other dataset hand trajectory.
- Keep EPIC-Contact central-frame labels as the highest tier. Propagated frames
  require the shipped quality fields and are retained only with an explicit
  confidence/validity mask.
- Convert both sides to the repository's metres/world-frame convention and
  emit the existing semantic hand-point contract plus 24 future object steps.
- Preserve the source object identity and `init_from_frame`; do not bridge a
  FoundationPose reinitialization or silently fill missing frames.
- This Probe may produce an engineering-ready pack only. It cannot establish
  pseudo-label accuracy or a learning gain.

## Bounds and stop conditions

Metadata and quality CSVs are kept under `tmp/ref7_epic_metadata_probe/`.
The first shard is capped at the advertised 146 MB; total new output is capped
at 5 GiB and the repository's 20 GiB free-space reserve remains in force.
No GPU is needed for the metadata/geometry audit, and no running GPU job may be
disturbed. Stop before downloading more shards if the first shard has no valid
overlap, if scale/frame conventions cannot be established, or if the bridge
would require inventing a missing hand or object trajectory.

## Current evidence

The EPIC-Contact quality tables contain 2,035 train clips over 124 videos and
237 test clips over 29 videos. ObjectForesight's object-level train/val index
contains all 124/29 videos at the video-ID level. Mapping through the official
EPIC action annotations found 1,388 candidate action-ID overlaps; the first
shard tested (`P01_03`) had no matching action trajectory. A train/train exact
candidate is now available at `P03_03_4054_4116_right_plate` ↔
`P03_03_23`, with all 30 Contact source rows inside the 75-frame
ObjectForesight pose range and `init_from_frame=47`, `num_reinits=0`.

The bounded EPIC-Contact conversion wrote a candidate pack for this clip at
`outputs/cm-pointflow-effect-pretrain/epic-contact-overlap-20261007/epic_contact_candidate_p03_03/`.
It preserves the 11 semantic right-hand points and a deterministic 512-point
object surface sample. The object canonical reconstruction is internally
stable (maximum per-frame RMS drift below 3e-7 m), all source hand/object rows
pass finite/rigid checks, and all quality rows are marked high confidence.
The source frame clock has 2/3-frame gaps at about 59.94 Hz; there are no
exact 4+24 windows under the strict contiguous-source rule, while explicit
time resampling yields four candidate windows and marks four gap-crossing
targets. The candidate remains `training_allowed=false` because EPIC-Contact
stores per-frame camera coordinates and this pack has no camera-to-world
extrinsics.

The independent ObjectForesight audit confirms source-frame overlap but does
not pass a direct raw-GLB/pose injection: the released TRELLIS mesh and
`T_c_o` require a separate scale/coordinate calibration. The test-only
`P03_13_1926_2103_right_bottle` example is sparse (mostly 6-frame gaps), has
only 2/3 high-confidence rows, and does not qualify as a 30-Hz training
candidate. Full ObjectForesight is about 0.86 TiB and outside the current
300 GB campaign cap, so this remains a bounded probe.

For the train/train `P03_03` pair, the video/mask review identifies the same
red plate, so the current blocker is not an obvious object-identity mismatch.
The pose contracts still disagree quantitatively: Contact's `object.cam_t`
starts near `[−0.123, 0.238, 5.162]` and ends near `[0.010, 0.135, 6.427]`,
whereas ObjectForesight `T_c_o` is near `[−0.057, 0.210, 0.642]` and ends
near `[0.087, 0.216, 0.649]`. Applying the declared c2w SpaTracker
extrinsics preserves this discrepancy; a single global scale/translation fit
still leaves about 0.40 m RMS center residual. Thus ObjectForesight's camera
trajectory cannot currently calibrate Contact's metric labels.

## Decision note

The immediate decision is whether to train from the converted EPIC-Contact
candidate. The evidence supports the same-clip hand/object identity and local
mesh reconstruction, but not yet the stationary-world temporal contract:
strict source-clock windows are empty, resampling crosses explicit gaps, and
the available ObjectForesight camera/pose scale does not calibrate Contact's
metric trajectory. Therefore this Probe is `UNCLEAR`: keep the candidate for
conversion review, do not start training, and obtain the original EPIC camera
calibration or validate an explicit anchor-camera contract before freezing a
native manifest. Until then, keep EPIC-Contact as a validation/calibration
source and do not expand ObjectForesight pairing.
