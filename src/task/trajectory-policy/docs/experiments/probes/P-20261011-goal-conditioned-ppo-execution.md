---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-conditioned-ppo-execution
experiment_id: P-20261011-goal-conditioned-ppo-execution
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 90e78dd5e6c9c074330b87e0eda4982aea9c27b5
claim_id: C3
hypothesis_family: HF-trajectory-policy-execution-interface
probe_index_in_family: 6
seed_pool: probe
seeds: [314]
decision_changed_if_positive: retain the direct-action Stage 1 PPO variant and compare it against a residual/nominal adapter
decision_changed_if_negative: stop the direct-action Stage 1 route and obtain an explicitly training-eligible teacher or introduce nominal-plus-residual control before further training
status: UNPROMISING
run_id: goal-conditioned-ppo-execution-20261011-r1
---

## 2026-10-22 有效性边界（本注记优先于下方历史外推）

- **结论处置：降级。** 保留该 final actor 的局部物理失败；方法与 v6 解码瓶颈的推断降级，r1 仅留工程记录。
- **能支持：** 可审计 r2 的 16×542 无干预执行 hold45/long-held 均 0/16，接线检查覆盖项通过；r1 角色标记有误，不能替代 r2 身份。
- **不能支持：** 不能支持 v6/τ 或 PPO 普遍无效，也不能把本波接线 PASS 当完整物理资格或唯一机制诊断；单 motion/seed 与固定预算限制外推。

本卡不得再被引用为“v6/τ 无效”的证据。历史数值、状态、门槛和运行产物原样保留；
此处处置限定其当前可引用范围，不重写旧实验。501 轨迹评分、UNKNOWN 准入、voxel 与日志
修复的影响按实际执行链归属，不能无证据套用到本卡所有旧运行，也不能倒灌成修后结果。
共同事实、逐卡处置和复现入口见[历史卡片卫生报告](../../../../../../docs/audit/historical-card-hygiene-20261022.md)。


# Frozen evaluation of the Stage 1 PPO actor

## Motivation and decision note

The preceding PPO continuation completed its bounded training budget but training
sampling had no long hold. This separate frozen execution Probe prevents the
training trace from being treated as a success claim. It reuses the same 16-env
native setup and goal-only hand/object packet as the BC evaluation, loads only the
final PPO actor, and records all measured geometry and dispatch values.

The actor receives the unchanged live-state plus eight-step hand/object-goal
contract. It has no future q, teacher command, phase, clock, force label or Cm
input. The comparison is to the earlier self-owned tracker calibration, not to
the engineering-only packet's forbidden actions.

## Fixed inputs and gate

* native input manifest:
  `outputs/consequence-evaluator/hand-execution-inputs-contact-capture-20261010-r1/inputs.json`;
* goal-only packet:
  `outputs/consequence-evaluator/ref7_2-same-cpu-teacher-contact-packet-20261010-r1.pkl`;
* geometry/reset contract:
  `outputs/consequence-evaluator/ref7_4-tau-geometry-20261010-r1/`;
* frozen PPO actor:
  `outputs/trajectory-policy/goal-conditioned-ppo-20261011-r1/final.pt`;
* one idle GPU, 16 environments and 542 controls, with a 300-second timeout.

The gate is at least 12/16 long-held terminal episodes and native clipping below
1%, matching the BC execution Probe. This remains a single-motion Probe and does
not establish generalization, REGRIND reproduction or Cm utility.

## Results

The valid run completed in
`outputs/trajectory-policy/goal-conditioned-ppo-execution-20261011-r2/` in
53.1 seconds on GPU0. All 542 controls completed in each of 16 environments;
peak Torch allocation was 149,659,648 bytes (about 7.54 GiB native process
memory), with no early reset and no native clipping.

The deterministic PPO actor reached `0/16` long-held terminal episodes,
`0/16` 45-frame holds and median held frames `0`, so the predeclared gate is
`UNPROMISING`. The first r1 execution produced the same physical result but had
an incorrect `goal_bc` role label in its manifest; it is retained as a metadata
setup failure and r2 is the auditable run.

The independent audit
`outputs/trajectory-policy/goal-conditioned-ppo-execution-audit-20261011-r1/audit.json`
passed feature reconstruction (`8.57e-3` max), deterministic policy replay
(`7.34e-7` max), requested/applied equality, finite PD targets and goal-only
reference authorization. Thus the frozen direct-action PPO failure is not a
current execution-wiring failure.

## Limitations and follow-up

Direct native-action BC/PPO is closed for this motion under the fixed budget. The
next meaningful test is a nominal-goal plus bounded residual adapter or a fresh
rollout marked `training_allowed=true`; high-level PointWAM/ACT forecasting
remains deferred.

## 2026-10-22 负结果 CPU 审计：结论范围修订

有效可审计物理产物是r2（90e78dd5），卡头run_id的r1不能替代它；r1记录/归档不足归工程问题。保留r2 hold45/long-held均0的单motion现象和已验接线PASS；不否决PPO一般方法，更不证明WAM v6解码是瓶颈。

本追加不改变原始数字、gate或运行产物；完整身份、版本边界与排除项见[逐卡证据审计](../../../../../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-card-audit-20261022.md)。
