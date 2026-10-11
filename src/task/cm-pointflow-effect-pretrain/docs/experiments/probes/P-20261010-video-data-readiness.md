---
schema: ref2dex.probe.v2
probe_id: P-20261010-video-data-readiness
experiment_id: P-20261010-video-data-readiness
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: b550e841dd0511e7f012d8d3e5cc788e0efc2569
claim_id: C1
hypothesis_family: HF-video-data-readiness
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: qualify a bounded weak-video adapter before mixed-source training
decision_changed_if_negative: retain native main sources and acquire denser video hand labels before training
status: UNPROMISING
run_id: video-data-readiness-20261010-r1
---

# Local video data readiness after semantic contract repairs

Class: Decision (data audit, no model training).

Question: Do the already-local ObjectForesight/EPIC-Contact pairs contain any
complete H4+K24 hand windows once the original source split, all 11 semantic
joints, source quality, exact video origin, and bracketed interpolation are
honored? This serves the Task's point-dynamics pretraining subgoal; it does not
change Mission C1 or test causal robot usefulness.

Hypothesis: The previous positive conversion window may survive conservative
label qualification. A surviving **train** window permits a small scene adapter
engineering probe; zero surviving windows blocks training from this local pair
subset and prioritizes denser labels. Camera reprojection alone is insufficient.

## Protocol and decision note (before run)

- Read metadata only from the three already-local shards (previous inventory:
  34 scenes) and match the two local Contact clips by source video/frame overlap.
- Honor ObjectForesight train/val and Contact train/test separately. A Contact
  test record never becomes train through ObjectForesight membership.
- Use metadata `start_frame` without subtracting padding again; require decoded
  video/depth lengths to equal `stop_frame-start_frame`. Record 59.94 Hz source
  and 29.97 Hz stride-two clocks honestly; do not call these native 30 Hz data.
- Candidate starts advance two source frames. Each window contains 28 target
  frames. Require valid side, `is_valid`, all selected semantic joints and finite
  coordinates at both interpolation endpoints, verified/high-confidence quality,
  source frame gap <=3, and no extrapolation. Thresholds are fixed before run.
- Replay the first candidate window of the two pairs with corrected masking.
  Failed LK identities are permanently invalid; background points exclude
  current hand/object masks. Count >=16 moved points valid over the entire window
  separately from background coverage. Never infer training permission from it.
- Cost: <=15 min CPU wall time, <=128 MB new outputs, no data download, no GPU,
  no training, no source modification or checkpoint overwrite. CPU is appropriate
  for metadata/label/geometry checks, not a historical model-computation policy.
- Stop on unexpected source/schema mismatch or output collision; preserve failed
  outputs. Positive: investigate clock/decoder adapter. Negative: keep native
  OakInk2/GRAB/ARCTIC training sources and address video-label coverage first.
  No new external authorization is needed for this bounded local audit.

Inputs: original repository
`outputs/cm-pointflow-effect-pretrain/epic-contact-overlap-20261007/` (read only).

Outputs: current worktree
`outputs/cm-pointflow-effect-pretrain/video-data-readiness-20261010-r1/`.
Tools: `tools/audit/audit_epic_video_readiness.py` and
`tools/audit/convert_epic_scene_flow.py` relative to this Task.

## Results

Initial committed run completed: zero qualified hand windows. Two geometry
replays retained 176/74 moved points valid throughout the first window but only
17/0 valid right-hand frames respectively, so neither emits an H4+K24 window.
The initial inventory reported only 20 scenes because it counted scene clock
metadata; the P01_03 shard has 14 additional clips without `action.meta.json`.
The inventory correction records those missing clocks explicitly and treats
video/time overlap as a candidate match rather than confirmed object pairing.
The initial `readiness.json` is preserved; rerun inventory goes to
`readiness-inventory-r2.json`, with its own execution commit/hash.

Result: UNPROMISING for immediate joint hand/scene training from this local
EPIC subset; this does not test the video-pretraining method. Inventory rerun
at `a7783600bd80aca743dddb6b539ae0c829483829` finds34 local scenes,20 scene-clock
metadata files and14 missing clocks, all in P01_03. Two Contact files have four
source-video/time overlaps: P03_03_23 (11 starts), P03_13_11 (28), P03_13_12 (52),
P03_13_13 (too short, zero starts). All95 starts yield zero qualified windows.
These overlaps are temporal candidates, not four verified same-object pairs.

| First-window replay | plate / P03_03_23 | bottle / P03_13_12 |
| --- | --- | --- |
| Metadata source start |4054|1929|
| Source split (ObjectForesight / Contact) |train / train|train / test|
| Qualified source right-hand rows |20/30|14/30|
| Qualified target right-hand frames |17/28|0/28|
| Fully valid moved-point tracks |176|74|
| Candidate H4+K24 windows |0|0|
| Training allowed |false|false|

Both geometry replays ran converter commit `b550e84` (SHA256
`d3712efc666869938417eddfc694d8f10fc0741acf092b709e8cb96c076e04c6`).
The bottle source starts before the video origin; it may supply a bracketed
observation, but cannot shift the video to the first Contact row. Its five/six
frame label gaps do not satisfy the fixed <=3-frame rule. Historical side-only
window counts are superseded for training qualification.

Decision: retain OakInk2/GRAB/ARCTIC as main training sources. Do not start mixed
EPIC training or spend on a rigid-native decoder adapter for zero accepted
windows. Next video work should test video-only track prediction separately or
obtain denser, confidence-masked hands; both need qualified source clocks and
their own adapter. The [tactile schema probe](P-20261010-egotouch-label-schema.md)
checks a cheaper auxiliary-supervision entry point.

Verification:18 EPIC/tactile contract tests pass, including original failed-LK
revival reproduction (`tmp/video-readiness-regression/repro.json`), joint/quality
masking, non-extrapolation, source split preservation, missing-clock inventory
and frame-identity preservation. No model training or checkpoint write occurred.

## Limitations / future evidence

Missing dense hand labels is a data blocker, not a negative result for video
pretraining. RGB LK status does not establish stable physical surface identity;
forward/backward or teacher consistency needs separate qualification. Hand world
placement still uses contemporaneous scene-object depth and shares label errors;
it cannot support independent hand/object causality claims. No rigid-native
decoder adapter, tactile model or robot policy comparison is tested here.
EgoTouch acquisition/pressure-clock sampling remains a separate subsequent gate.
