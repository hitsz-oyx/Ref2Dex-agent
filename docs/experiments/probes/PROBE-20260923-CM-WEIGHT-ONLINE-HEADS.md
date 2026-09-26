# Cm PPO 权重分量在线 Probe

probe_id: `PROBE-20260923-CM-WEIGHT-ONLINE-HEADS`

date: 2026-09-23

branch: `agent/cm-weight-mechanism`

## Question and decision

V1.52 的 aligned Cm 权重优于置乱；离线 Probe 发现接触与预测位移均有
真实一步事件排序信息。在线收益主要需要哪一个分量？

若 effect-rank 接近 joint 且显著好于 contact-rank，下一路线优先简化为
效应头；反之优先接触头。若 joint 同时优于两者，保留联合/交互结构。
若两 seed 方向不一致，本轮标 `UNCLEAR`，不直接进入论文 ablation。

## Minimal protocol

沿用 V1.51 同一自训练 s3 e260 checkpoint、V1.37 冻结 CmLite、
64env、seed70、horizon32、minibatch256、学习率1e-5 和相同 reward/reset
配方，继续到固定 e300。只新训两臂：`contact_rank` 与 `effect_rank`。
每个 rollout step 保留 `joint` 在真正参与 PPO actor 的活跃样本上的
**精确权重多重集**，按对应单分量预测分数排序重新分配；非活跃样本权重不变。
这让各臂的活跃边际分布/均值/计算量相同，只测试权重-样本匹配依据。
原 joint/off 使用已冻结 e300 checkpoint，不重新挑选 epoch。

先 unit test 和 e262 工程 smoke，确认有限性、mode 日志与 checkpoint 恢复；
随后两臂训练到 e300。仅新 seed129、130，每臂各一次64环境严格完整首
episode，共 8 次评估，不按中途结果追加 seed。对照只作方向判断，
不计算正式置信区间，也不升级最终 Cm utility claim。

## Budget and stop condition

GPU5/6 最多同时两张，使用前必须检查无他人进程；预计训练每臂<10min、
评估总计<15min，新增产物<5GB。源码/输入漂移、GPU 冲突、非有限权重、
检查点合同不符或任一严格评估失败时停止，不拼接结果。

## Result

Status: PROMISING（仅方向性；两个 seed、每臂每 seed 一次，不是 Validation）

e262 两种新模式 smoke 均 `COMPLETED`，日志中的 mode、有限权重与 checkpoint
恢复正常。两臂均从同一个自训练 e260 源固定训练至 e300；8 次新 seed 严格完整
首 episode 评估及源码指纹检查全部 `COMPLETED`，没有官方 actor checkpoint。

| seed | joint | effect-rank | contact-rank | off |
| ---: | ---: | ---: | ---: | ---: |
| 129 | 51/64 | 42/64 | 22/64 | 37/64 |
| 130 | 32/64 | 48/64 | 23/64 | 35/64 |
| 合计 | 83/128 | **90/128** | 45/128 | 72/128 |

位移排序相对 off 在两个 seed 均为正（+5、+13），接触排序均为负（−15、−12）。
joint 与 effect-rank 在两个 seed 方向相反（+9、−16），不判断谁更强。
这是短 Probe，且各策略训练后轨迹分布不同；结果不证明预测位移的精确因果机制。
父 manifest：`outputs/CmResidual/agent_cm_weight_heads_probe/run_manifest.json`。
新训练 e300 SHA256：contact-rank
`1128223798bb5905ae53d5c7eae7ddb0bc78d83242cfffcc972b03581959a42c`；
effect-rank `12bb9f921cc95fe240f69185118b34c9028b944b90ec9727e88eb3af261f1607`。

## Decision update

不优先继续优化纯接触概率排序，也不宣称 effect-rank 优于 joint。下一最小问题是：
effect-rank 的好处是否需要**当前状态下的真实动作**？做保持同一步活跃权重
多重集、但把动作在活跃样本之间打乱后再计算效应排序的训练 Probe。
若不需要动作对应关系，当前结果更像状态显著性加权；若需要，则更支持
动作条件一步效应表示。随后才值得扩大 matched Validation。
