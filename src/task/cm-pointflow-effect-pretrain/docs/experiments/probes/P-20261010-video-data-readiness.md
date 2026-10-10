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
status: UNCLEAR
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

## Limitations / future evidence

Missing dense hand labels is a data blocker, not a negative result for video
pretraining. RGB LK status does not establish stable physical surface identity;
forward/backward or teacher consistency needs separate qualification. Hand world
placement still uses contemporaneous scene-object depth and shares label errors;
it cannot support independent hand/object causality claims. No rigid-native
decoder adapter, tactile model or robot policy comparison is tested here.
EgoTouch acquisition/pressure-clock sampling remains a separate subsequent gate.
