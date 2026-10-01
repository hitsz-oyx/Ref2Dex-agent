# D-20261001-after-hf08-value-audit

日期：2026-10-01；决策 owner：root。用户已授权 Mission-level 持续推进，期限为
2026-10-03 23:59（Asia/Shanghai）。

**问题。** HF06–HF08 的固定接法没有形成可升级的正向信号。停止局部调参后，
哪项最小检查能决定动作条件物理预测参与策略训练的下一步？

**关键证据。** [HF08 完整 Probe](../experiments/probes/P-20260930-cm-physical-value.md)
已经完成 48/48 评价；原生 gate 为 `UNPROMISING`，终点持续保持为 Cm 33/384、
plain PPO 41/384、direct Q 40/384。V 是新建并预训练的网络，之后还在线更新了
160 个 epoch；标签为混合 reward 的 return，不能称为成功概率。公共带噪池仅有
15/1920 个稳定成功 episode。现有 source 开发误差不能证明当前策略 V 校准，
也不能证明未执行动作的排序。同 source e0 重复分数不一致，完整初态未记录；
这需要限制解释，但分数波动本身不自动否定总体 matched 比较。历史同策略重复、
状态恢复及完整 episode action-value 检查已做，不原样重开。

**行动与理由。** 保持 Cm policy utility 为 `OPEN`，HF08 的固定实现不升级
Validation、不继续调参。通过 Broker 派发
`T-20261001-hf08-value-target-audit` 给固定 `agent_cm`，复用原生数据做
CPU-only 价值目标与标签审计，查看 reward 分量、独立成功 episode 和分层误差。
这能区分实现错误、目标关系薄弱与需要当前策略数据的情形，比先重训 V 或再跑
PPO 便宜。不得凭 loss 或 source MC 宣称当前策略 V 已收敛。

**成本、分支和停止条件。** 审计上限 2 CPU 线程、20 分钟、1 GiB；不采集、不
训练、不用 GPU。输入 hash 漂移、非有限数据、episode/terminal provenance 缺失
或预算到限就停止并报告剩余不确定性。若有明确实现错误，先由执行角色修复；
若需要当前策略完整轨迹，另派有界采集给 `agent_rl`；若证据不足以判别，不用
增加拟合步数替代诊断。审计交付后重新选择能改变路线的最小任务；任何新的
科学 Probe 必须有独立机制、预算和预设判据，不能重置 HF08 已用完的 slot。

**外部授权边界。** 当前行动在既有授权内，无额外授权需求。改变核心研究问题
或 claim、突破 CAMPAIGN 资源限制、不可逆外部操作或新增长期角色时再请求授权。

**首轮审计验收。** worker 已完成交付，但 root 暂不接受其完整报告：脚本把
30 步 tracker 事件标成 `stable_success`，与主指标 45 步不一致；在线 V 一节
只读取了 Cm dynamics 的开发诊断，没有直接检查保存的 V 参数。折扣分量显示
held 项约占源池 return 的 73%，不能将该池描述为主要由模仿项主导。报告的
分层成功统计必须由 owner 修正，在线 V checkpoint 必须直接核查；在此之前
不以该报告启动重训或新科学 Probe。原始交付保留，修正作为独立有界 CPU 任务。
修正任务为 `T-20261001-hf08-value-audit-label-repair`，计算上限 10 分钟、2 CPU
线程、1 GiB。首轮脚本若已响应 CONTROL 改动须说明，R1 的旧报告与 Broker hash
保留为审计记录；root 只验收新的 R2 交付。

**可并行的工程准备。** `agent_rl` 只准备当前策略完整轨迹采集接口和 CPU
合同检查，暂不启动仿真。基线是已有 physical-value collector；新接口限定两个
保存的 Cm-value e420 actor，96 个环境的完整首 episode、三 motion 均衡、帧 0
起始，共同诊断 seed 290。必须读取 HF08 原生配置：其
`enable_eps_greedy=false`，因此行为是 Gaussian 采样后按原合同裁剪，不是
deterministic actor mean，也没有 source 采集池的额外动作噪声。保留相同 reward、
reset 前 next-state、terminal/timeout、previous-action 和动作采样 provenance。
准备上限 2 CPU 线程、20 分钟、0.1 GiB；无 GPU、采集或训练。采集是否启动仍
等待 R2 验收和独立有界运行派发，不增加或重置 HF08 的 policy-utility Probe slot。

**R2 验收及并行分工。** root 已核对修正交付的原生标签、checkpoint hash 和
逐文件导入合同，接受 R2；R1 仅保留为被修正的历史依赖。源池主指标成功为
15/1920，holdout 仅 1/384 个成功 episode；两个 e420 的 V 参数均有实际更新，
optimizer step 为 7680。这证明 V 被训练过，尚不能判断是否充分或适用于当前策略。
`agent_rl` 继续实现真实采集 runtime 及 CPU 合同检查，暂不执行仿真；
`agent_cm` 并行实现完整 episode 的 MC 与冻结 PPO critic 的 32-step GAE(lambda)
目标诊断，并以合成完整轨迹验证终点 mask 和边界；`agent_infra` 同步已验收事实与
任务账本。后两项分别限 2 CPU/20 分钟/0.1 GiB 和 2 CPU/10 分钟/0.1 GiB。
当前只授权工程实现和文档维护，不以合成数据形成科学判断。采集器与诊断器
验收后，另派固定 actor 的有界当前策略轨迹诊断，区分 V 对在线目标拟合不足与
在线 bootstrap 目标偏离完整回报；frame-0 诊断子分布不代表全部训练 reset 分布。
依据结果选择下一机制，不预定重训 V，不重新开放 HF08 调参预算。
