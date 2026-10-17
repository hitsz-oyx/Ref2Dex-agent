---
schema: ref2dex.probe.v2
experiment_id: P-20261017-gt-tau-new-sequence
task: trajectory-policy
branch: main
git_commit: 886f481b
status: UNPROMISING
classification: Decision
hypothesis: The frozen retargeter R driven by oracle future hand plan GT tau can still stably grasp and hold an unseen sequence.
question: Does GT tau preserve the physical grasp/hold gate on a new sequence outside the historical s3 airplane physical run?
decision: A pass permits moving to a learned executor on this sequence; a fail blocks that route and triggers implementation/data review before PPO.
sequence: s1_airplane_lift
sequence_path: /home2/wyy/oyx_ws/Ref2Dex/data/processed_data/dexplore_reconstructed_v120_coordfix_v4/converted_attempt1/s1_airplane_lift/interaction_hand_inspire.pt
model_sha256: 414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f
protocol: 8 official teacher, 8 source-PD control, 8 GT-tau-R oracle episodes; controls equal to motion frames minus one at 30 Hz; frozen official DExplore simulator and checkpoint.
pass_gate: GT-tau-R hold45 >= 6/8, success >= 6/8, clipping_rate < 1%, median_held >= 0.8 * source-PD median_held.
resource: local GPU 0 for teacher and GPU 1 for evaluation, pinned runtime /home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train; coexist with pre-existing <=0.4 GiB processes.
outputs: outputs/trajectory-policy/gt-tau-new-sequence-20261017-r4/
scientific_scope: This is an oracle GT-tau physical qualification only; it does not qualify a learned executor, WAM, tokenizer, or PPO policy.
run_id: gt-tau-new-sequence-20261017-r4
target_decision: UNPROMISING
runner_status: UNCLEAR because the legacy full-task calibration rule also requires source-PD placement >=6/8; source-PD placement was 5/8 while its hold gate was 8/8.
---

## Decision note

- Current decision: verify the only immediate blocker, whether GT tau remains physically stable on a new sequence.
- Key prior evidence: historical s3 physical gate passed for GT tau/R on 8 held-out environments; the current new-sequence gate is still unmeasured.
- Chosen action: run the bounded local oracle physical probe with the frozen audited `tau.pt` on the coordinate-fixed DExplore conversion. The earlier r1/r2 attempts are retained as invalid execution attempts: r1 exposed a fixed-length runner defect, and r2 used a non-DExplore-native `partial_filtered` tensor whose object frame was displaced.
- Cost and stop condition: one 32-environment teacher source plus one 24-environment evaluation, each bounded to 300 s; stop after this gate and do not start tokenizer/PPO work from this result.
- External authorization: local GPU use is explicitly authorized by the user; no other processes will be killed or modified.

## Result

The coordinate-fixed teacher calibration completed on GPU0 with 32/32 hold45 and 24/32 controlled-placement successes (median held 114 frames, clipping 0%). The evaluation on GPU1 used 8 teacher, 8 exact source-PD, and 8 GT-tau/R environments. Teacher reached 8/8 hold45 and 8/8 success; source-PD reached 8/8 hold45 and 5/8 success; GT-tau/R reached 0/8 hold45 and 0/8 success, with median held 6.5 frames and clipping 0.174%.

For the stated grasp/hold decision this is `UNPROMISING`: GT-tau loses the hold gate on a coordinate-valid new sequence even though teacher and source-PD both hold. The runner's conservative full-task label remains `UNCLEAR` solely because source-PD placement calibration was 5/8. Independent replay [audit](../../../../../../outputs/trajectory-policy/gt-tau-new-sequence-20261017-r4/gt-tau/audit.json) passes with zero error for actual PD, measured-state features, GT-tau features, model output, and requested PD.

The earlier r1/r2 attempts are retained but excluded: r1 used a fixed 542-step runner on a 279-frame input and reset at the natural terminal frame; r2 used the non-DExplore-native `partial_filtered` tensor (object z 1.337 m versus 0.918 m in the coordinate-fixed conversion), so its 0/32 teacher result is invalid input evidence.
