# Direct collector repair and factual V diagnostic

- 问题：修复已确认的 collector logstd/sigma API 错误，取得冻结当前策略完整轨迹，区分 factual V 对冻结 PPO lambda 目标的误差及 lambda/MC 差异。
- 授权：用户明确要求“修复采集器然后诊断V”，随后明确“不要交给子代理，你直接接管他们的工作”。root直接实现与运行。
- 证据：R3b 在 Normal(std=raw logstd) 处失败，零行；已有修复遗留文件被 root 复核，实际两个 checkpoint + native CommonPlayer 对照及合同共22测试通过。外置观测归一化原路径已经执行；不沿用旧错误归因。
- 行动：同一 diagnostic run r4，新独占 task-owned output；1GPU/1200秒/2GiB/192000行；固定两策略各96完整首回合，无训练。所有旧失败成本保留，新预算独立报告，HF08 PAUSED/1of1不变。
- 成功后：CPU2线程/20分钟评价真实历史上的保存V、PPOcritic、MC、冻结GAE32，按checkpoint/motion/success/drop分层；不把单轨迹MC误差升级为条件期望失准或收敛结论。
- 失败后：保留失败证据，仅在明确工程原因与剩余预算内修复，不更改seed/任务/目标以凑结果。任何资源或数据合同异常立即停止。
- 外部授权：当前边界内无新增外部权限需求。
