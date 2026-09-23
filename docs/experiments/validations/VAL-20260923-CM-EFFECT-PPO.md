# Validation: 动作条件 Cm 效应排序能否稳定改善 PPO？

validation_id: `VAL-20260923-CM-EFFECT-PPO`

date: 2026-09-23

branch: `agent/cm-effect-validation`

status: PRE-REGISTERED

## Claim and decision

检验窄范围 claim：在 s3 单轨迹、固定自训练 e260 源和冻结 CmLite 下，
训练期使用真实动作的一步预测竖直位移对 PPO actor 活跃样本排序加权，
是否跨训练 seed 稳定优于 matched Cm-off，且优于打乱动作对应关系的对照。
若两者都成立，保留 effect-rank 作为候选主方法并再研究稳定性/跨轨迹；
否则停止将当前配方当作已证实的 Cm 方法，转入 Cm 表征或接法重设计。

## Treatment and matched controls

三臂均从同一自训练 DExplore s3 e260 actor checkpoint 恢复，不使用官方 actor。
训练 seed71–74，每 seed 三臂，固定到 e300；只比较固定末 epoch，不按训练奖励
挑 checkpoint。64env、horizon32、minibatch256、学习率1e-5、相同运动输入、
奖励、reset 课程、PPO actor/critic 网络和真实环境 reward/advantage。

- `effect_rank`：冻结 V1.37 CmLite 用当前状态与记录动作预测 |world dz|，
  按该分数为活跃样本分配每步 joint 权重多重集。
- `action_shuffled`：每步在活跃样本间打乱动作，再对原状态预测 |world dz|；
  用打乱动作的分数分配**同一规则生成的**活跃 joint 权重多重集。
  独立 RNG，不消耗策略采样 RNG。
- `off`：普通 PPO actor 样本权重 1，不调用 Cm。

effect-rank 和 action-shuffled 保持每步活跃 joint 权重多重集与模型相同；
只改变状态-动作-权重的对应关系。两臂在训练分布分化后，每步多重集不要求
跨臂数值逐一相同。最终评估三臂均只加载普通 actor，不调用 Cm。

冻结输入 SHA256：e260 actor
`16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`；
CmLite `396f5e0c92ecf67b2a096568c1a538b7c86d40779a594729f786df6c64ed204a`；
s3 corrected_r2 manifest
`2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038`。
训练开始前冻结代码 commit 与关键源码/运动 tensor 指纹，运行期间检查漂移。

## Evaluation and primary metric

全新评估 seed133–138，每个训练 seed × 评估 seed × arm 运行一次，
共 72 次；每次 64 环境、关闭提前终止、严格只计完整首 episode。
每臂总计 4×6×64=1536 次。主要指标为 `lift_success` 比率。
不同 arm 不按 env_id 配对；匹配单位是同训练 seed 与评估 seed。

主效应 `D_off = effect_rank - off`。动作信息效应
`D_shuffle = effect_rank - action_shuffled`。先完成固定矩阵，再算结果；
不按中途成败调整 seed、系数、epoch 或方法。以训练 seed 与评估 seed
两个维度独立重采样 20,000 次，报告两个差值的双侧 95% percentile CI。

## Predefined gates

`Cm policy utility` 通过需同时满足：`D_off >= +8pp`、4 个训练 seed
各自汇总差值 >0、双维 bootstrap 95% CI 下界 >0。

`action-conditioned information` 通过需同时满足：`D_shuffle >= +8pp`、
4 个训练 seed 各自汇总差值 >0、双维 bootstrap 95% CI 下界 >0。

只有**两个门都通过**才将本配方的窄范围 claim 标为 `SUPPORTED`。
若 `D_off <= 0` 或 `D_shuffle <= 0`，当前配方标为 `REFUTED`；
其余未过门情况标为 `INCONCLUSIVE`，不包装为证明。另报告 effect-rank
24 次运行中每次是否达到 ≥58/64；这只是单次抓取稳定性门，
即使主门通过也不能自动声称稳定抓取。

## Budget and stop conditions

最多同时 GPU5/6 两张，运行前与每对任务前检查占用；绝不干扰其他进程。
预计训练与评估合计 <2h、新产物 <10GB、总产物远低于 300GB。
源码/输入指纹漂移、GPU 冲突、非有限训练、checkpoint/strict episode
合同失败时停止并记录 `INVALID_IMPLEMENTATION` 或运行失败，不拼接矩阵。
无科学结果驱动的提前停止。

## Result

run_status: NOT_STARTED

conclusion: PENDING

Evidence: PENDING

Limitations: 只测试 s3 单轨迹与固定 e260 起点；即使通过，也不证明跨轨迹、
跨手、原版 Cmv2 架构作用或独立的最终抓取稳定性。
