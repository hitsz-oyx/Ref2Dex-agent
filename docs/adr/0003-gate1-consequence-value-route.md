# 先验证真实后果到长期价值，再训练 Cm

日期：2026-10-03

## 决策

将 `docs/ref.md` 定义的 Gate 1 置于整手控制原语、Cm 预测和蒸馏之前。Gate 1 使用真实 simulator rollout 中的 effect `E`、interaction consequence `I` 和 return-to-go `G`，比较 `V_H`、`V_HEI` 与 `V_HAEI` 的 held-out 预测能力。

Gate 1 的主要目标保留当前 PPO 的 reward 与折扣定义，同时报告成功、连续保持和掉落等任务结果。数据按完整 episode/context 分组切分，不能按逐帧随机切分；Probe 结果不升级为正式科学结论。

## 原因

旧的固定八候选 oracle 已达到其有限动作银行的 8/12 观察容量，继续在该动作集内拟合 Cm 不能提供新的因果信息。直接进入控制原语或蒸馏会把表示、价值建模和动作执行混在一起。先验证真实 `E+I` 是否能解释长期价值，可以区分表示本身无效、Cm 预测损失和策略接入失败三种瓶颈。

## 边界

Gate 1 的真实未来只用于监督与离线特权对照，不能作为部署 actor 的输入，也不构成 Cm policy utility 证据。`E` 是物体未来位姿/线速度/角速度；`I` 是手部刚体相对物体的位姿/速度和可用接触量，并显式保留有效性 mask。Gate 1 之前不做 Cm 训练或蒸馏。

## 退出条件

如果 `V_HEI` 相对 `V_H` 的主要 held-out 误差下降达到预注册的 10% 且 episode-cluster bootstrap 区间排除 0，同时 `V_HAEI` 相对 `V_HEI` 没有同等量级的增益，则标记为 `PROMISING` 并进入 Gate 2。否则标记为 `UNPROMISING` 或 `UNCLEAR`，先复盘数据合同和 consequence 表示，不扩大训练预算。
