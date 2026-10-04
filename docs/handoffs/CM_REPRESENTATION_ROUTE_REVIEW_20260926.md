# Cm 表征路线只读复盘（2026-09-26）

Owner：`agent_cm_temporal` / `agent_Cm/selective`；完成 turn
`01a0de41-97ee-7380-97f9-24dadf746d5e`。本次只读复核没有修改子代理分支、运行实验或消耗 GPU。

## 决策问题

训练期 Cm 表征是否构成区别于旧 auxiliary 配方、值得立即启动的新高层 C3 假设？

## 关键证据

- [3D auxiliary Probe](../experiments/probes/P-20260924-cm-ppo-aux-representation.md)：Cm-on 136/256，Cm-off 121/256，探索性差异 +5.86pp，低于预声明 +8pp 升级门；没有置乱 target 或通用辅助任务 placebo。
- [H10 12D auxiliary Probe](../experiments/probes/P-20260924-cm-h10-ppo-aux-representation.md)：Cm-on 143/256，Cm-off 133/256，差异 +3.91pp；一个训练 seed 为负，未过稳定性门。
- 两条路线都已测试以冻结 Cm target 监督 PPO shared actor representation。换 target 维度没有建立稳定 policy gain。[D008](../archive/2026-10-04-research-governance/RESEARCH_DEBT.md) 的 placebo/多 seed 对照仍处于 deferred 状态。
- HF01–HF05 的失败只限制各自配方，不能推出 Cm 普遍无效；C3 保持 `OPEN`。

## 建议

保持当前冻结，不登记 HF06。推理时显式使用的 causal bottleneck 在结构上可以区别于旧 auxiliary，但目前缺少能预先固定唯一 target 的机制证据；直接启动会成为高成本 target 搜索。若以后有明确 target，再以独立 goal 和 Decision Checkpoint 比较 matched Cm-on、Cm-off、置乱 bottleneck 与通用辅助 placebo。本次复核不是新 Probe 或 Validation。

## 主代理终态审查

- 注册 thread `01a0d7cf-3a82-7a93-b05c-ffecb361f036` 的 rollout 记录本轮 `task_complete`：turn `01a0de41-97ee-7380-97f9-24dadf746d5e`，耗时约 98 秒，交接状态为 `COMPLETED/READ_ONLY_DECISION_MEMO`。注册 Goal 的 `blocked` 属于此前 HF05 实验 Goal，不代表本轮只读任务未完成。
- 本轮工具调用仅为 `git show`、`git grep`、`git status`、`git rev-parse` 和文档/代码读取；没有训练、fit、collector、评估或文件修改调用。子代理工作树保持干净，HEAD 为 `0d4abd721dbb88224cf9bea1308c81084e889d5a`。
- 主代理复核了两份原始 auxiliary Probe 卡：3D 为 136/256 对 121/256（+5.86pp，低于 +8pp 门），H10 为 143/256 对 133/256（+3.91pp，且一个训练 seed 为负）。这支持“不升级旧接法”的建议，不能证明 Cm 普遍无效。
- 审查时归属 GPU PID 为空；HF05 旧 manifest 的修改时间和分支 HEAD 均未变化。poller 已记录本轮 turn ID 并向当前主代理 thread 发送一次完成事件。没有可集成的子代理提交。

审查结论：接受只读交接；维持当前冻结和 C3=`OPEN`。未来只有提出可预先固定的不同机制并完成新的 Decision Checkpoint，才考虑独立 Probe。
