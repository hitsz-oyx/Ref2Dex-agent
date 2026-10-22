---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-conditioned-ppo
experiment_id: P-20261011-goal-conditioned-ppo
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: d07b739be20921cc533f469e9802ca510f03bf23
claim_id: C3
hypothesis_family: HF-trajectory-policy-execution-interface
probe_index_in_family: 5
seed_pool: probe
seeds: [313]
decision_changed_if_positive: continue Stage 1 with a frozen PPO low-level actor and run a separate frozen evaluation
decision_changed_if_negative: stop this direct-action continuation and obtain a fresh training-eligible teacher or add an explicit nominal/residual adapter
status: UNCLEAR
run_id: goal-conditioned-ppo-20261011-r1
---

## 2026-10-22 有效性边界（本注记优先于下方历史外推）

- **结论处置：加注。** 加注保留 UNCLEAR 训练证据，不升级成 PPO、预训练或从零 RL 的方法否决。
- **能支持：** 实际 24 次有限 PPO 更新、49,152 训练交互和 actor 改变，可作为训练工程证据；原 UNCLEAR 保留。
- **不能支持：** 不能支持冻结评估成功率、PPO 收敛或 v6/τ 无效。训练 reward 和少量 episode held 不替代独立评估，后续 r2 失败也仅约束单 motion 短预算策略。

本卡不得再被引用为“v6/τ 无效”的证据。历史数值、状态、门槛和运行产物原样保留；
此处处置限定其当前可引用范围，不重写旧实验。501 轨迹评分、UNKNOWN 准入、voxel 与日志
修复的影响按实际执行链归属，不能无证据套用到本卡所有旧运行，也不能倒灌成修后结果。
共同事实、逐卡处置和复现入口见[历史卡片卫生报告](../../../../../../docs/audit/historical-card-hygiene-20261022.md)。


# Stage 1 goal-conditioned PPO continuation

## Motivation and decision note

The goal-conditioned BC fit has low validation action error but its frozen physical
execution loses hold in the first few controls. The independent audit shows that
the native executor is wired correctly and that the student leaves the training
feature manifold after its first action. Ref2 explicitly places PPO after BC, so
this bounded continuation tests whether measured-state on-policy updates can
repair that covariate shift while preserving the same goal-conditioned contract.

The actor starts from the BC network. It samples a bounded native action through
an active-coordinate tanh mask. The measured reward contains current hand/object
goal tracking, physical proximity/lift/hold/loss and action-rate terms. It does not
read future q, teacher commands, phase, clock, contact labels or Cm values. This
is the direct-action Stage 1 variant; it does not silently claim REGRIND's
joint-reference gains or residual architecture.

## Fixed inputs and budget

* native input manifest:
  `outputs/consequence-evaluator/hand-execution-inputs-contact-capture-20261010-r1/inputs.json`;
* goal-only reference packet:
  `outputs/consequence-evaluator/ref7_2-same-cpu-teacher-contact-packet-20261010-r1.pkl`;
* BC initialization:
  `outputs/trajectory-policy/goal-conditioned-bc-20261011-r3/best.pt`;
* geometry/reset contract:
  `outputs/consequence-evaluator/ref7_4-tau-geometry-20261010-r1/`;
* one idle GPU, 16 environments, 24 updates of 128 controls, and a 480-second
  timeout. The run writes fresh outputs and logs GPU utilization, memory and ETA.

The PPO run is engineering/training evidence only. It is not a frozen evaluation
and cannot establish grasp success. A positive training result requires a separate
frozen execution Probe with the same observation contract; a negative result stops
this direct-action variant and motivates a nominal/residual adapter or a fresh
training-eligible teacher rollout.

## Results

The bounded run completed in `outputs/trajectory-policy/goal-conditioned-ppo-20261011-r1/`
at code commit `d07b739`, seed 313, 24 updates, 3,072 controls and 49,152
environment transitions. GPU0 stayed available with about 7.56 GiB native
memory and the Torch peak was 177,190,400 bytes. All 24 PPO updates passed the
finite/behavior-logprob/KL checks; the joint KL stayed below `0.015`, actor
parameter change was `0.0529`, and mean rollout reward was `0.1073`.

This is `UNCLEAR` training evidence: five 542-control episodes completed during
the run, but their maximum held frames were at most 3 and none was terminal held.
The final actor therefore required the separate frozen evaluation below.

## Limitations and follow-up

This single-motion, single-seed Probe does not test high-level forecasting,
generalization, failure-bin sampling, matched Cm-on/off training, or the full
REGRIND reference-tracking configuration. The frozen evaluation is recorded in
`P-20261011-goal-conditioned-ppo-execution.md`; direct-action BC/PPO is closed
for this motion if that gate also fails.

## 2026-10-22 负结果 CPU 审计：结论范围修订

保留d07b739b实际24更新/49152训练交互及actor变化；训练return不是成功率，也不是独立冻结评估。本卡UNCLEAR不应升级为PPO方法负结果；另卡r2确定性评估失败只约束这个短预算/单motion策略，不能推断预测预训练或从零RL无效。

本追加不改变原始数字、gate或运行产物；完整身份、版本边界与排除项见[逐卡证据审计](../../../../../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-card-audit-20261022.md)。
