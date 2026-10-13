---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-ref27-rollout
experiment_id: P-20261013-goal-point-wam-ref27-rollout
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: c15d8a7b
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 38
seed_pool: probe
seeds: [292]
decision_changed_if_positive: retain C candidate for separately qualified manipulation comparison if autonomous lift45 occurs
decision_changed_if_negative: inspect measured closed-loop divergence and preserved baselines; do not infer Cm useless from one rollout
status: UNCLEAR
run_id: wam-ref2-execution-C-final-20261013-r1
---

# C final native rollout: can the trained actor lift and hold?

Result: Pending four-step engineering smoke and one full542-control rollout.
Decision: User explicitly requests physical simulated execution of final C.

## Motivation and Decision Note

Decision Probe for the manipulation subgoal: does offline C adaptation produce
an autonomous stable lift, or does measured-state closed-loop drift persist?
The answer changes whether to retain this candidate for further manipulation
comparison or inspect drift. Cheapest useful test is one fixed seed292/frame0
airplane reset, teacher and exact GT-tau qualified Chunk-R calibration rows,
and the C final actor row. This is a development Probe, not matched Cm-on/off
causal Validation or a random-initialization/generalization test.

Use immutable Probe37 step50500 decoder/optimizer packet only as a frozen
actor decoder overlay on its exact original full WAM and Probe32 warm
scene/goal/hand-query patch. Use the exact qualified Probe34 Chunk-R fixed
in C training. Old native bridge's MLP-R path remains the default; explicit
C/Chunk arguments select the new path. Checkpoint identity, module keys,
finite tensors, frozen-state history, native units and peer identities must
agree. No extra training, teacher intervention, recovery takeover, hand
readout change or reward/weight search.

Actor queries use only own live H4 + planned object endpoint t+24; recorded
future hands feed only GT calibration row. Chunk predicts8 commands and
executes only first at each tick, passive action columns0. Native command
clamping/PD conversion unchanged and captured from the actual simulator.
First-query actor-only vs direct full-WAM equivalence is checked live.

One GPU2, two owned pinned-runtime peers, <=600s group, outputs+scratch<=1GB,
global outputs+tmp<=400GB. Four-step smoke first; then542 controls/543 states,
30Hz ~18.07s simulated. Stop on source/input drift, foreign GPU process,
nonfinite output, wrong frame/native conversion, unexpected reset or deadline.
Fresh immutable output directories, protected input checkpoints and external
Ref2Dex read-only. No new branch or remote push.

## Fixed outcome before execution

Legacy grasp requires hand-object sampled gap<=10mm, object unsupported by
table and height>=30mm over initial pose; exclude reset frame. Stable lift is
>=45 consecutive held frames (~1.5s). Report any brief lift, maximum continuous
hold, height, clipping, later loss and full episode controlled-placement result
separately. Calibrate local grasp if teacher and current GT-Chunk-R each hold
>=45 frames with <1% clipping. If calibration fails or wiring invalid, actor
failure is UNCLEAR, not a negative method result. Autonomous success uses
actor row only; all official-teacher actions are confined to calibration row.

Actual simulator cameras record three panels at30fps; video is observation,
not the success metric. Independent saved-trace audit checks actual commands,
first-of8 selection, live feature frames, planned goals and outcomes.

## Inputs and outputs

- C patch: `outputs/goal-point-wam/wam-effect-ddp-20261013-r3/train/checkpoint_step50500.pt`
- Full base: `outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt`
- Warm policy: `outputs/trajectory-policy/wam-ref2-joint-multitakeover-20261012-r1`
- Qualified R: `outputs/trajectory-policy/wam-v2-control-fit-20261012-r1/chunk.pt`
- Smoke: `outputs/trajectory-policy/wam-ref2-execution-C-smoke-20261013-r1`
- Scientific rollout: `outputs/trajectory-policy/wam-ref2-execution-C-final-20261013-r1`

## Limitations / future evidence

One deterministic development reset, no seed suite or perturbation test.
Full task and sustained-lift criteria remain distinct. Any lift success does
not attribute gains to Cm: matched effect-off training and further physical
Validation remain deferred. No additional rollout or training required merely
for completeness; a valid failure does not close the full C representation.
