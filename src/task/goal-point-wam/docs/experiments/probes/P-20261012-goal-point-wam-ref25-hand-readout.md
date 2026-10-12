---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref25-hand-readout
experiment_id: P-20261012-goal-point-wam-ref25-hand-readout
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 35
seed_pool: probe
seeds: [295, 296]
decision_changed_if_positive: retain explicit PTv3 hand readout and consider staged teacher-forced/frozen predicted-tau world consistency
decision_changed_if_negative: stop C and retain the qualified Chunk-R while preserving this short readout comparison
status: UNCLEAR
run_id: wam-hand-readout-fit-20261012-r1
---

# ref2_4 B after ref2_5: explicit observed-hand PTv3 readout

## Decision Note

Probe34 control-aware Chunk-R qualifies GT execution (74held vs oldR75,
teacher73; clip0). Under the staged ref2_4/ref2_5 authorization, test whether
explicit hand readout preserves or improves forecasts. This **Decision Probe**
serves manipulation; goal32 endpoint/memory, raw adapter and PTv3 stay.

Old and explicit-hand arms load the exact Probe32 policy, initialization295/
draws296, sequentialGPU2, each200updates. Gather current full-packed PTv3
features by raw semantic metadata14/15/16 and scene_object<0 into2x11x128;
invalid pointszero, duplicates rejected. Independently of<=256 scene sampling,
sum with raw-hand query residual. No new parameters; separate subclass keeps
historical keys and base model source unchanged.

Both arms use fixed qualified Probe34 Chunk-R, parameter-frozen with actor
input gradients. Loss: actual tau MSE/.05 plus the same first/tail/PD control
objective used to qualify R. Nominal16TRAIN and259 actual expert recovery
windows8/16/32; no pure failed futures. Batch64=16startup/16later/32expert
(11/11/10 per source). AdamW, scene1e-5, other policy1e-4, clip1, eval mode;
statistics buffers, R, action adapter/world decoder fixed. No world loss or
predicted-tau feedback in B; existing teacher-forced world interface preserved.
No from-scratch WAM, extra goals, 3PoinTr or RL.

Report h1/first8/all24, first-PD XYZ p95, clipping on nominal64 fixed windows,
startup64 and three actual recovery TRAIN pools. Goal sensitivity is the same
cached-scene response to +/-20mm endpoint x translations (not physical accuracy).
Predeclared non-regression per panel: anchored h1/first8<=1.1old+.05mm,
firstPD<=1.1old+.1mm, clipping<=old+.01; sensitivity>=max(.001mm,.5old).
All must pass before C. A short fit cannot rule out the readout architecture.

Resource supplement: paired1200s/20GPUmin, fresh wam-hand-readout outputs/
scratch<=500MB, global400GB. Separate from cumulative wam-v2 R80MB because
each scene/goal policy packet is207MB. Keep two final packets, no intermediate
saves. Stop on nonfinite, semantic/data scope error, frozen drift, unmatched
draws, foreign GPU process, deadline or size. No new branch/external boundary.

## Implementation / evidence

- `src/task/goal-point-wam/src/goal_point_wam/hand_anchored.py`
- `src/task/goal-point-wam/tests/test_hand_anchored.py`
- `src/task/trajectory-policy/tools/run/probe_wam_hand_readout.py`

Two tests pass: shuffled packed points keep semantic identity, scene sampling
cannot remove gathered hand gradients, invalid tokenszero, duplicates rejected.
Engineering smoke uses1update/arm, saves no policy. Scientific run fixed200.

## Results

Pending matched offline comparison and independent metric/label reconstruction.

## Limitations / future evidence

Recovery pools are TRAIN; nominal episodes share deterministic motion/reset.
This is exploratory, not Validation or autonomous grasp. Both B arms have
identical objective/R but differ from earlier Probe32 training, so only the
contemporaneous arm controls the interface. C Stage1/2 needs a separate
protocol after B qualifies; this card does not silently enable world feedback.
