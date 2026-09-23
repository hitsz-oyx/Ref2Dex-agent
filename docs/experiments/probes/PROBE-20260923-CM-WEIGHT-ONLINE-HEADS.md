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

Status: PENDING

Key evidence: PENDING

## Decision update

PENDING
