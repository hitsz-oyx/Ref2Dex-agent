# Cm 表征路线只读复盘（2026-09-26）

Owner：`agent_cm_temporal` / `agent_Cm/selective`；完成 turn
`01a0de41-97ee-7380-97f9-24dadf746d5e`。本次只读复核没有修改子代理分支、运行实验或消耗 GPU。

## 决策问题

训练期 Cm 表征是否构成区别于旧 auxiliary 配方、值得立即启动的新高层 C3 假设？

## 关键证据

- [3D auxiliary Probe](../experiments/probes/P-20260924-cm-ppo-aux-representation.md)：Cm-on 136/256，Cm-off 121/256，探索性差异 +5.86pp，低于预声明 +8pp 升级门；没有置乱 target 或通用辅助任务 placebo。
- [H10 12D auxiliary Probe](../experiments/probes/P-20260924-cm-h10-ppo-aux-representation.md)：Cm-on 143/256，Cm-off 133/256，差异 +3.91pp；一个训练 seed 为负，未过稳定性门。
- 两条路线都已测试以冻结 Cm target 监督 PPO shared actor representation。换 target 维度没有建立稳定 policy gain。[D008](../RESEARCH_DEBT.md) 的 placebo/多 seed 对照仍处于 deferred 状态。
- HF01–HF05 的失败只限制各自配方，不能推出 Cm 普遍无效；C3 保持 `OPEN`。

## 建议

保持当前冻结，不登记 HF06。推理时显式使用的 causal bottleneck 在结构上可以区别于旧 auxiliary，但目前缺少能预先固定唯一 target 的机制证据；直接启动会成为高成本 target 搜索。若以后有明确 target，再以独立 goal 和 Decision Checkpoint 比较 matched Cm-on、Cm-off、置乱 bottleneck 与通用辅助 placebo。本次复核不是新 Probe 或 Validation。
