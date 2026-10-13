---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-ref27-rollout
experiment_id: P-20261013-goal-point-wam-ref27-rollout
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: 615a3b42
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 38
seed_pool: probe
seeds: [292]
decision_changed_if_positive: retain C candidate for separately qualified manipulation comparison if autonomous lift45 occurs
decision_changed_if_negative: inspect measured closed-loop divergence and preserved baselines; do not infer Cm useless from one rollout
status: UNPROMISING
run_id: wam-ref2-execution-C-final-20261013-r1
---

# C final native rollout: can the trained actor lift and hold?

Result: COMPLETED542 controls; final C actor never lifts/held0, while teacher/GT-Chunk hold434/441 frames. Current checkpoint rollout is UNPROMISING.
Decision: Preserve the valid negative rollout; inspect contact-region closed-loop drift before further scaling. This does not refute C or establish Cm causal value.

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

Visualization reconstructs measured hand points and object poses at30fps;
it is labeled measured-state reconstruction, not actual camera footage. Independent saved-trace audit checks actual commands,
first-of8 selection, live feature frames, planned goals and outcomes.

## Inputs and outputs

- C patch: `outputs/goal-point-wam/wam-effect-ddp-20261013-r3/train/checkpoint_step50500.pt`
- Full base: `outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt`
- Warm policy: `outputs/trajectory-policy/wam-ref2-joint-multitakeover-20261012-r1`
- Qualified R: `outputs/trajectory-policy/wam-v2-control-fit-20261012-r1/chunk.pt`
- Smoke: `outputs/trajectory-policy/wam-ref2-execution-C-smoke-20261013-r3`
- Scientific rollout: `outputs/trajectory-policy/wam-ref2-execution-C-final-20261013-r1`

## Limitations / future evidence

One deterministic development reset, no seed suite or perturbation test.
Full task and sustained-lift criteria remain distinct. Any lift success does
not attribute gains to Cm: matched effect-off training and further physical
Validation remain deferred. No additional rollout or training required merely
for completeness; a valid failure does not close the full C representation.


## Engineering startup

Smoke r1 failed camera creation before any control: headless disabled graphics.
Owned camera config enabled graphics in r2. It completed4 controls and saved
camera video/result, but the native process exited SIGSEGV (-11) during final
cleanup; original logs preserved. This is a graphics lifecycle failure, not
an actor grasp negative. Full scientific execution returns to the verified
no-graphics runtime (615a3b42); no model/physics/loss change. State-based
visualization is generated offline after rollout, explicitly not camera video.
Smoke r3 COMPLETED with both peers exit0 in51.88s, actual4 controls/5states,
first actor-only vs full-forward equivalence checked. C step50500 and qualified
Chunk checkpoint identities match across peers; all commands/PD finite and
native conversion checks pass. Four interface/admission tests passed.


## Full rollout result

Scientific r1 completed542 controls/543 measured states with native and server
exit0, group128.89s, seed292/GPU2, execution code615a3b42. Server first-query
actor-only vs direct full-WAM discrepancy0; C patch step50500 exact and Chunk
checksum same as training/GT qualification. No teacher intervention in actor.

| Actual row | Longest held frames | Stable hold seconds | Maximum height above reset | Clipping | Full controlled placement |
| --- | --- | --- | --- | --- | --- |
| Teacher calibration | 434 | 14.47 | 522.4mm | 0% | true |
| GT tau + qualified Chunk-R | 441 | 14.70 | 506.1mm | .9225% | false |
| C final WAM + same Chunk-R | 0 | 0 | 0mm | 9.2251% | false |

The actor never satisfies a held frame, never lifts above reset; the object
stays supported and settles. This is not a successful pickup followed by loss.
Teacher and GT-Chunk establish the declared local grasp calibration (hold45
and clipping<1%). GT drops during placement, intermediate loss1 and terminal
height-.7114m; full-task calibration in native result is false. Do not conflate
its good grasp calibration with full-placement qualification.

Independent NumPy audit recomputes actual applied commands, independent native
PD conversion including half finger scaling/couplings, only first-of8 execution,
passive columns0, clipping, own live R state, GT current-frame tau, own measured
H4, planned goal, and every stored outcome. All10 numerical checks maxdifference0;
independent consecutive held counts agree. CPU is appropriate for pure array/
statistics auditing; model execution stayed on GPU. No fresh training/inference
sweep or alternate checkpoint selection after seeing the failure.

Measured same-trace actor-vs-teacher wrist deviation: t24=7.706mm,
t32=7.213mm, t48=174.213mm, t56=393.974mm, t64=572.389mm. Corresponding
first-PD XYZ teacher difference t32=27.482mm, t48=353.660mm, t56=464.712mm.
Actor hand-object sampled gap at t48=155.637mm, t56=399.272mm; teacher first
held tick56/GT54. Divergence accelerates around the contact transition despite
initial first-PD difference.709mm. First actor clippingtick92, first128 clip6.25%.
These are descriptive drift diagnostics, not proof of the cause or an oracle
that entered the actor. Offline teacher-window accuracy has not established
closed-loop grasp robustness. Later actor wrist drift reaches several meters.

GPU peak observed18780MiB (native PhysX ~18160/inference594), utilization20..41%
with synchronous IPC, median server48.75ms/native124.26ms per step; no OOM,
nonfinite or timeout. GPU2 released after normal exits. Visualizations use the
recorded actual hand points/object poses, with a fixed workspace and explicit
out-of-view markers; they are reconstructed point geometry, not camera video.
The early camera attempts are engineering failures and are excluded from the
scientific result. No second scientific rollout, refit, DAgger or RL was run.

- [Native result](../../../../../../outputs/trajectory-policy/wam-ref2-execution-C-final-20261013-r1/native/result.json)
- [Independent audit](../../../../../../outputs/trajectory-policy/wam-ref2-execution-C-final-20261013-r1/native/root_effect_rollout_audit.json)
- [Drift diagnostics](../../../../../../outputs/trajectory-policy/wam-ref2-execution-C-final-20261013-r1/native/drift_diagnostics.json)
- [Measured-state video](../../../../../../outputs/trajectory-policy/wam-ref2-execution-C-final-20261013-r1/visualization-r2/measured_rollout.mp4)
- [Physical curves](../../../../../../outputs/trajectory-policy/wam-ref2-execution-C-final-20261013-r1/visualization-r2/outcomes.png)

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/trajectory-policy/tools/run/launch_wam_native_bridge.py --output outputs/trajectory-policy/wam-ref2-execution-C-final-20261013-r1 --controls 542 --experiment-id P-20261013-goal-point-wam-ref27-rollout --head-dir outputs/trajectory-policy/wam-ref2-joint-multitakeover-20261012-r1 --chunk-dir outputs/trajectory-policy/wam-v2-control-fit-20261012-r1 --effect-checkpoint outputs/goal-point-wam/wam-effect-ddp-20261013-r3/train/checkpoint_step50500.pt
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/trajectory-policy/tools/audit/audit_wam_effect_rollout.py --native outputs/trajectory-policy/wam-ref2-execution-C-final-20261013-r1/native
```

Completed scientific output+scratch6.76MB<1GB; reconstructed movie181 frames/
18.10s at10fps, source sampled every3 ticks from the30Hz measured trace.
Four interface tests and repository changed-file verification pass.
