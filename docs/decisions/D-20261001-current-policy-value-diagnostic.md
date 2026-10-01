# D-20261001-current-policy-value-diagnostic

日期：2026-10-01；owner：root；分类：Decision，回顾性价值诊断。

**问题。** HF08 的 V 实际更新过，但源池 holdout 只有一个主成功 episode。
在当前策略分布上，后续优先处理 V 拟合/分布适配、bootstrap 目标，还是返回
物理表示及策略接法？关键证据见已验收的
[HF08 R2](../handoffs/HF08_VALUE_TARGET_AUDIT_R2_20261001.md)。

**选择与最便宜的判别。** 先验收采集器和 CPU 诊断合同；随后另行通过 Broker
派发一次冻结策略的完整首 episode 采集，两个已有 cm_value e420 actor 各 96
环境、三 motion 各 32、frame 0、共同 seed 290。沿用原 reward、Gaussian 行为
和动作裁剪，无额外噪声、teacher 或参数更新。该面板只代表明确的 frame-0
诊断子分布，不代表训练的全部 reset 分布。记录同一动作 forward 的原 PPO
critic 与正确 vector-step rollout 相位，终点按原实现 done mask 截断。

**读数及下一步。** 对 factual 完整轨迹比较保存的 V、原 PPO critic、完整 MC
与冻结 critic 的 32-step GAE(lambda) 诊断目标，按 episode 汇总并保留 motion
和主成功/drop 分层。GAE 诊断目标不是恢复的历史训练标签；单条 realized MC
具有随机性，误差不单独证明条件期望失准或 bootstrap 偏差。若 V 对诊断目标
仍明显失配，优先核查适配/优化；若 V 跟随诊断目标但二者与完整回报有明显差异，
优先核查目标构造和 bootstrap；若这些不足以解释失败，返回物理表示与策略接法。
这些是选择下一项最小检查的线索，均不自动授权 HF08 局部调参或形成正式结论。

**成本与停止。** 由 agent_rl 执行未来采集，整个双 checkpoint 面板共享最多
1 GPU、1200 秒、2 GiB 和 192000 行；初始化、检查与导出计入运行预算。
CPU 分析由 agent_cm 承担。输入漂移、非有限值、无法完成首 episode、归属冲突
或任一预算到限即停止并保留 FAILED/INCOMPLETE 证据；稀有成功分层不据此下结论。
当前只派发 CPU 工程准备；实际仿真须采集器和诊断器验收通过后单独派发。

**边界。** HF08 仍 PAUSED、utility Probe slot 1/1 已用；本次不重训、不换 seed
重扫收益、不升级 Validation，不重置任何 family 预算。Cm policy utility 仍 OPEN。
后续新科学 Probe 须区别于已经失败的局部方案，另记机制、预算和判据。
本诊断在现有 Mission/Campaign/用户授权内，无新增外部授权需求。
