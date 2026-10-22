---
schema: ref2dex.probe.v2
probe_id: P-20261017-gt-tau-new-sequence
claim_id: C3
hypothesis_family: HF-trajectory-policy-executor-transition
probe_index_in_family: 2
seed_pool: probe
date: 2026-10-17
decision_changed_if_positive: A pass permits moving to a learned executor on this sequence.
decision_changed_if_negative: A fail blocks that route and triggers implementation/data review before PPO.
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

## 2026-10-22 有效性边界（本注记优先于下方历史外推）

- **结论处置：降级。** 保留固定 R 的 hold 资格失败；完整 placement 结论降级，r1/r2 撤出方法比较。
- **能支持：** 仅坐标修正后的 r4：teacher/source-PD hold45 各 8/8，固定 GT-τ→R 为 0/8，说明该冻结 R 在这一序列未通过 hold 资格。
- **不能支持：** 不能支持 learned executor、v6/τ 或 PPO 无效；source-PD placement 5/8 未过完整校准。r1 runner 长度/reset 与 r2 partial_filtered 错坐标不能算方法结果，特权 GT 也未检验 learned WAM 预测。

本卡不得再被引用为“v6/τ 无效”的证据。历史数值、状态、门槛和运行产物原样保留；
此处处置限定其当前可引用范围，不重写旧实验。501 轨迹评分、UNKNOWN 准入、voxel 与日志
修复的影响按实际执行链归属，不能无证据套用到本卡所有旧运行，也不能倒灌成修后结果。
共同事实、逐卡处置和复现入口见[历史卡片卫生报告](../../../../../../docs/audit/historical-card-hygiene-20261022.md)。


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

## 2026-10-22 负结果 CPU 审计：结论范围修订

仅保留坐标修正后的 r4：teacher/source-PD 的 hold45 都是8/8，而 GTτ/R 为0/8。完整placement校准的source-PD仅5/8，原runner UNCLEAR不能当成已认证完整任务负结果；抓持代理缺失仍可保留。r1固定长度/自然终止reset与r2错误partial_filtered坐标均移出方法比较。本卡是固定R资格，不能归为learned executor、WAM v6或PPO失败。实际执行manifest c0c85956与卡头886f481b分开保留。

本追加不改变原始数字、gate或运行产物；完整身份、版本边界与排除项见[逐卡证据审计](../../../../../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-card-audit-20261022.md)。
