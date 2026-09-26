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

Status: PROMISING（动作对应关系；仅两 seed 方向性 Probe）

e262 smoke 和固定 e300 训练 `COMPLETED`；动作置乱臂 e300 checkpoint SHA256
`a612c4cf4acbee75d9367b2a4b6fa6775e17c01cb2f8874f157957dacd52de4e`。
6 次 strict/full-first-episode 评估、输入/checkpoint/源码指纹均通过，
无官方 actor checkpoint，最终推理均不加载 Cm。

| seed | 真实动作 effect-rank | 动作置乱 rank | off |
| ---: | ---: | ---: | ---: |
| 131 | 43/64 | 15/64 | 34/64 |
| 132 | 33/64 | 27/64 | 35/64 |
| 合计 | **76/128** | **42/128** | **69/128** |

effect-rank 对动作置乱两 seed 均为正（+28、+6），置乱均低于 off（−19、−8），
达到预设的方向性条件。effect-rank 对 off 一正一负（+9、−2），因此**未证明**
不同 seed 的稳定策略增益。此对照保持每步活跃权重多重集不变，但训练后状态
分布会分化，且仅单一训练 seed；不能升级为正式 Cm utility claim。
父 manifest：`outputs/CmResidual/agent_cm_effect_action_probe/run_manifest.json`。

## Decision update

动作条件一步位移信息值得保留为候选主方法；不再把纯接触概率排序当主路线。
现在进入 Decision Checkpoint：是对 effect-rank/动作置乱/off 做跨训练 seed 的
matched Validation，还是先重设计/适配 Cm 以提高 effect-rank 对 off 的稳定性。
