# Cm 效应头的动作对应关系 Probe

probe_id: `PROBE-20260923-CM-EFFECT-ACTION-ALIGNMENT`

date: 2026-09-23

branch: `agent/cm-weight-mechanism`

## Question and decision

effect-rank 在两个新 seed 上都优于 Cm-off，而 contact-rank 都落后。
但 effect-rank 可能仅仅按状态显著性加权，未必利用真实动作对应的一步模型信息。
本 Probe 问：打断状态与记录动作的对应后，效应头的在线收益还在吗？

如果保留真实动作的 effect-rank 在两个 seed 都优于动作置乱臂，且置乱臂
不超过 off，则优先把“动作条件效应”作为候选主方法并进入 matched Validation。
若置乱臂接近或优于 effect-rank，不能声称动作条件 Cm 信息是关键，改查状态
显著性权重或更强 action-ranking 设计。方向不一致则标 `UNCLEAR`。

## Minimal protocol

新训练一臂 `effect_action_shuffled_rank`：从同一个自训练 s3 e260 actor
继续到固定 e300，seed70、64env、相同 PPO/reward/reset/冻结 CmLite。
每个 rollout step 用真实动作计算 joint 权重；仅在 PPO actor 活跃样本之间
打乱动作，将打乱动作与原状态输入同一 CmLite，再按其预测 |dz| 给**同一活跃
joint 权重多重集**排序分配。独立 PyTorch RNG，不消耗策略采样 RNG。
与既有 effect-rank 的区别只有决定权重与样本匹配的动作对应关系。

e262 工程 smoke 后固定训练 e300，不选中间 checkpoint。评估仅新 seed131、132，
每 seed 的 effect-rank、action-shuffled-rank、off 各一次64环境严格完整首 episode，
共 6 次。旧 checkpoint 不重训，不加 seed，不作为正式统计验证。

## Budget and stop condition

GPU5/6 最多两张，启动前查他人进程；预计新增<3GB、wall time<20min。
输入/源码漂移、GPU冲突、有限性/恢复/strict episode 合同失败时停止。

## Result

Status: PENDING

Key evidence: PENDING

## Decision update

PENDING
