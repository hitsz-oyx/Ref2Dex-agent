---
schema: ref2dex.probe.v2
probe_id: P-20261018-direct-tau-dagger
claim_id: C3
hypothesis_family: HF-trajectory-policy-executor-transition
probe_index_in_family: 5
seed_pool: probe
experiment_id: P-20261018-direct-tau-dagger
task: trajectory-policy
branch: main
git_commit: 40d5d4b5
date: 2026-10-18
classification: Decision
status: UNPROMISING
hypothesis: Querying the official teacher on learner-visited unseen states and fine-tuning the repaired direct tau executor with the teacher's current-q PD target can repair the remaining closed-loop failure.
question: Does a DAgger-style learner-state correction improve direct tau execution on the same unseen s1_airplane_lift sequence?
decision_changed_if_positive: retain the reference tau plus live-state teacher-PD DAgger route as the next executor candidate before PPO.
decision_changed_if_negative: do not conclude that tau is unusable; distinguish action/executor contract limits from the need for a stronger interaction-conditioned policy.
protocol: Autonomous roll-in of the repaired r3 direct executor on 8 unseen direct environments. At each tick, query the official teacher on the same live direct observation, record its native action and current-q PD target, pair them with the independent reference tau for that source tick, then warm-start the r3 policy and train on equal original/query batches. Evaluation is a fresh frozen-policy run on the same unseen sequence with no teacher intervention.
query_semantics: reference tau remains tau_k = R_ref_t(h_ref[t+k] - h_ref[t]); teacher PD is computed from the official teacher native action and the live student q at the visited state. No future student hand is used as a tau label.
source: outputs/trajectory-policy/gt-tau-new-sequence-20261017-r4/teacher/trajectory.npz
base_checkpoint: outputs/trajectory-policy/direct-tau-bc-r3-20261018-r2/best.pt
gate: direct arm requires hold45 >= 6/8, success >= 6/8 and clipping < 1%; teacher is calibration only.
runtime: /home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train, Torch 2.0.1+cu118, GPU 0 with pre-existing context only.
---

## 2026-10-22 有效性边界（本注记优先于下方历史外推）

- **结论处置：加注。** 加注保留该一轮固定配方未恢复抓持的局部结论；不外推到未来轨迹重规划、其他执行器或 v6。
- **能支持：** 一轮 learner-state 当前动作/PD 查询可复算，离线 action L1 改善；无教师干预的本序列 learner hold45 为 0/8，teacher hold45 为 8/8、placement 为 7/8，满足本波旧校准。
- **不能支持：** 不能支持 v6/τ 无效或 DAgger 普遍无效。查询是当前状态动作/PD 标签，没有生成离轨状态的新未来 τ；同 motion 环境拆分不能认证任务泛化，producer r1 保存失败不计方法结果。

本卡不得再被引用为“v6/τ 无效”的证据。历史数值、状态、门槛和运行产物原样保留；
此处处置限定其当前可引用范围，不重写旧实验。501 轨迹评分、UNKNOWN 准入、voxel 与日志
修复的影响按实际执行链归属，不能无证据套用到本卡所有旧运行，也不能倒灌成修后结果。
共同事实、逐卡处置和复现入口见[历史卡片卫生报告](../../../../../../docs/audit/historical-card-hygiene-20261022.md)。


## Motivation and decision note

The repaired r3 executor has correct reference-frame tau wiring and temporal order,
but its BC policy achieved 0/8 hold45 on the unseen physical sequence. This probe
asks the narrower question suggested by DAgger: can the teacher supply the correct
current-state PD target after the learner leaves the teacher state distribution?
The cheapest test is one autonomous learner roll-in, one teacher query per visited
state, one warm-start correction, and one frozen re-evaluation.

The query is intentionally explicit. The source future hand supplies the independent
reference plan tau. The official teacher is queried on the live direct observation;
its native action is converted using the live q into a PD target and both values are
stored. This is a valid current-state PD label. It is not a claim that the teacher
has generated a new future hand trajectory from the off-track state.

## Planned audit

The roll-in must preserve the 16-row native layout: eight teacher calibration rows
and eight autonomous direct rows. The teacher action queried on direct rows must be
computed before the learner action is applied. The recorded PD target must exactly
reconstruct from the queried action, current q, and native controller offset/scale.
The query packet is split by environment, with five direct environments for fit and
three for selection. The base standardizer remains frozen. The corrected policy is
warm-started from the selected r3 checkpoint and receives equal original/query
batches; no PPO or C input is used.

## Results

The autonomous roll-in completed 431 controls for eight direct environments and
recorded 3448 learner-state teacher queries. The queried teacher action and its
current-q PD target reconstruct with maximum absolute error 0.0. Independent
recomputation of all logged reference tau values also has maximum absolute error
0.0. The first roll-in attempt reached the end of physics but failed while saving
the query packet because it sliced an already direct-only axis twice; it is kept as
an aborted producer and is not included in the result below. The corrected run is
`outputs/trajectory-policy/direct-tau-dagger-rollin-20261018-r2/`.

The base r3 policy's action L1 on the held-out query environments was 0.41245.
After warm-start DAgger correction it was 0.11906. The original sequence
validation L1 after correction was 0.06503, slightly worse than the r3 base
validation L1 0.04030. These are offline diagnostics only.

The frozen DAgger policy was then evaluated without teacher intervention on the
same unseen corrected `s1_airplane_lift` sequence. The direct arm achieved
`0/8 hold45`, `0/8 success`, median held frames 0, clipping rate 0, and endpoint
position error 0.8311 m. The teacher calibration achieved `8/8 hold45` and
`7/8 success`, with clipping rate 0. The direct arm therefore fails the physical
gate and this Probe is `UNPROMISING`.

Artifacts:

- roll-in and query labels: `outputs/trajectory-policy/direct-tau-dagger-rollin-20261018-r2/`
- warm-start fit: `outputs/trajectory-policy/direct-tau-dagger-fit-20261018-r1/`
- frozen execution: `outputs/trajectory-policy/direct-tau-dagger-execution-20261018-r1/`

The run records the teacher's live-state PD target correctly, so this negative
result is not caused by the earlier tau-coordinate bug or by using teacher C.
It does show that one-round current-state PD DAgger, with the present direct-action
executor and reference tau input, did not recover unseen physical grasping.

## Limits and next action

This is one roll-in seed, one held-out physical sequence, and one warm-start fit.
The query tau is the independent source reference plan; the teacher was queried
for current-state action/PD only, so this is PD DAgger rather than a full teacher
future-trajectory DAgger. The result does not refute tau as a high-level plan, but
it does not justify PPO yet. Before another training budget, inspect the direct
action/PD target contract and whether the executor needs a richer interaction or
phase input; retain this run as the current DAgger negative Probe.

## 2026-10-22 负结果 CPU 审计：结论范围修订

保留实际manifest44c8b7ea的一轮当前learner-state动作/PD查询与无干预执行：离线action L1改善而本序列hold45仍0/8。本波teacher placement7/8满足其完整校准，但仍仅一序列一轮拟合；query不是从离轨状态产生完整future τ的重规划。producer r1保存packet错误的失败不计作方法结果；同motion五env拟合/三env选择不是独立任务泛化。

本追加不改变原始数字、gate或运行产物；完整身份、版本边界与排除项见[逐卡证据审计](../../../../../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-card-audit-20261022.md)。
