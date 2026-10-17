---
schema: ref2dex.probe.v2
experiment_id: P-20261017-gt-tau-new-sequence
task: trajectory-policy
branch: main
git_commit: c1024d5862ecf120daa3e9940cddc696ffc2a7aa
status: RUNNING
classification: Decision
hypothesis: The frozen retargeter R driven by oracle future hand plan GT tau can still stably grasp and hold an unseen sequence.
question: Does GT tau preserve the physical grasp/hold gate on a new sequence outside the historical s3 airplane physical run?
decision: A pass permits moving to a learned executor on this sequence; a fail blocks that route and triggers implementation/data review before PPO.
sequence: s1_airplane_fly_1
sequence_path: /home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_rl_partial_filtered_20260905/s1_airplane_fly_1/interaction_hand_inspire.pt
model_sha256: 414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f
protocol: 8 official teacher, 8 source-PD control, 8 GT-tau-R oracle episodes; 542 controls at 30 Hz; frozen official DExplore simulator and checkpoint.
pass_gate: GT-tau-R hold45 >= 6/8, success >= 6/8, clipping_rate < 1%, median_held >= 0.8 * source-PD median_held.
resource: local GPU 0, pinned runtime /home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train; coexist with pre-existing <=0.4 GiB processes.
outputs: outputs/trajectory-policy/gt-tau-new-sequence-20261017-r1/
scientific_scope: This is an oracle GT-tau physical qualification only; it does not qualify a learned executor, WAM, tokenizer, or PPO policy.
---

## Decision note

- Current decision: verify the only immediate blocker, whether GT tau remains physically stable on a new sequence.
- Key prior evidence: historical s3 physical gate passed for GT tau/R on 8 held-out environments; the current new-sequence gate is still unmeasured.
- Chosen action: run the bounded local oracle physical probe with the frozen audited `tau.pt`, because it is the cheapest direct test of the stated goal.
- Cost and stop condition: one 32-environment teacher source plus one 24-environment evaluation, each bounded to 300 s; stop after this gate and do not start tokenizer/PPO work from this result.
- External authorization: local GPU use is explicitly authorized by the user; no other processes will be killed or modified.
