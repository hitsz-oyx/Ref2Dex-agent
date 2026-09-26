# Validation: 动作条件 Cm 效应排序能否稳定改善 PPO？

validation_id: `VAL-20260923-CM-EFFECT-PPO`

date: 2026-09-23

branch: `agent/cm-effect-validation`

status: COMPLETED

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

run_status: COMPLETED

conclusion: REFUTED（当前 effect-rank 配方的联合正向主张；不否定所有 Cm）

## Fixed-matrix result

代码 commit `c5108d4`；训练 12/12、严格完整首 episode 评估 72/72 全部
`COMPLETED`，无被剔除运行。父 manifest 在每对任务前后核对关键源码、运动
tensor、模型/源 checkpoint 输入指纹；GPU 5/6 最多各一进程。

| 训练 seed | effect-rank | action-shuffled | off | effect−off | effect−shuffled |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 71 | 280/384 | 165/384 | 217/384 | +63/384 | +115/384 |
| 72 | 324/384 | 293/384 | 189/384 | +135/384 | +31/384 |
| 73 | 168/384 | 287/384 | 203/384 | −35/384 | −119/384 |
| 74 | 110/384 | 304/384 | 180/384 | −70/384 | −194/384 |
| 合计 | **882/1536 (57.42%)** | **1049/1536 (68.29%)** | **789/1536 (51.37%)** | **+93/1536 (+6.05pp)** | **−167/1536 (−10.87pp)** |

主效应 effect−off 为 +6.05pp，双维训练/评估 seed bootstrap 95% CI
**−14.71 至 +26.69pp**；4 个训练 seed 仅 2 个为正，未达到预设
≥8pp/每训练 seed 正/CI 下界正的 utility 门。因此当前配方的稳定
Cm-on/off 增益是 `INCONCLUSIVE`，不能作为论文证明。

动作对应效应 effect−action-shuffled 为 −10.87pp，95% CI
**−42.06 至 +20.18pp**；4 个训练 seed 仅 2 个为正。按预注册的
联合判定（点估计≤0），当前“真实动作对应的 effect-rank 优于打乱动作”
主张标为 `REFUTED`。区间跨零意味着**不能反向声称**动作置乱显著优于
effect-rank；这里只能说本验证不支持正向动作对齐主张。

effect-rank 的 24 次评估只有 **1 次**达到预设的单次稳定抓取门
≥58/64；稳定抓取目标失败。训练末期两种 Cm 权重均值大致都在 1.41–1.51，
仅凭权重均值不能解释跨训练 seed 的巨大差异。

## Decision and limitations

遵守已选 Option A 的失败分支：**停止继续扩大当前 effect-rank 配方的
评估，不将早期 Probe 的正向结果升级为结论**；下一研究路线应转向
Cm/接法的重设计或分布适配，并先做廉价 Probe。

action-shuffled 在此矩阵总体比 off 高，但这是非预注册的后验比较；
它仍通过真实动作预测生成每步权重多重集，只是打乱样本分配，因此不能
称作“完全无动作信息”或证明随机权重优于 Cm。需要新实验才能解释。
它也不同于 V1.52 的全样本权重置乱：这里仍保留原状态输入，可能保留
状态显著性信息；两个结果并不构成直接矛盾。

父 manifest：`outputs/CmResidual/cm_effect_validation/run_manifest.json`；
完整矩阵、每 seed 差值、CI 和判定：
`outputs/CmResidual/cm_effect_validation/analysis.json`；
child logs/checkpoints 见父 manifest。没有使用官方 actor checkpoint。

Limitations: 只测试 s3 单轨迹与固定 e260 起点；即使通过，也不证明跨轨迹、
跨手、原版 Cmv2 架构作用或独立的最终抓取稳定性。
