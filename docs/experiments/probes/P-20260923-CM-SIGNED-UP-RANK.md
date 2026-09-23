# Probe: Cm 有向一步效应能否改善 PPO 样本排序？

probe_id: `P-20260923-CM-SIGNED-UP-RANK`

date: 2026-09-23

branch: `agent/cm-postvalidation-diagnosis`

status: RUNNING

## Question and decision

正式验证否定当前 `abs(predicted world dz)` 排序的稳定增益。问题是冻结 Cm
是否已经在失败策略的状态上失准，还是把上升和下降都排成高权重损害了 PPO 接法。
这是 Decision Probe：如果有向排序在一个已失败训练 seed 上出现明显正信号，
才进入跨训练 seed 测试；否则不继续同一加权族，转向更不同的 Cm 接法。

## Cheapest diagnostic already completed

固定 V1.37 CmLite，对训练 seed74 的 effect-rank 与 action-shuffled e300
策略在同一评估 seed139 采 64env 严格首回合转移；再对 seed71 的
effect-rank 策略同样采样。三次仿真均 `COMPLETED`，不使用官方 actor。

模型并非完全失准：seed74 effect-rank 策略接触样本上，预测与真实 signed dz
Pearson r=0.664；真实动作的接触 EPE 3.86mm，打乱动作 EPE 5.30mm。
然而按旧 `abs(pred dz)` 评分的最高十分位，实际平均 signed dz = -5.06mm，
其中下落≥3mm占16.0%；按 `clamp(pred dz / 3mm,0,1)` 排序，最高十分位
实际平均 +4.01mm，下落占1.4%。seed71 的旧绝对值最高十分位同样平均
-3.82mm，而有向分数最高十分位平均 +8.72mm。

这些是训练后 on-policy 关联，不是同状态反事实，也不能解释训练差异的因果。
它们仅表明“有向分数”值得一个很小的在线 PPO Probe。

诊断产物：`outputs/CmResidual/cm_postvalidation_diagnosis/`；仿真转移位于
各 `outputs/Dexplore/cm_effect_val_t{71,74}_*/eval_s139_e300_full_cmdiag/`。

## Hypothesis and minimal online protocol

H1: 保持当前每步活跃 joint 权重多重集和所有 PPO/环境设定，只将分配权重的
排序分数从 `|pred dz|` 改为 `max(pred dz,0)`，可以避免强化预测下落样本，
在已失败训练 seed74 上优于现有 effect-rank，且至少不逊于 Cm-off。

Alternative: 离线有向性没有转化为策略收益；这种 actor 样本加权族不值得
继续局部调节。

从同一自训练 e260 checkpoint 训练 seed74 至 e300，唯一新 arm
`signed_up_rank`；Cm 冻结，仍只参与训练 actor 样本权重，真实 advantage
决定梯度方向，评估不用 Cm。对既有 seed74 effect-rank/off 的相同评估
seed133、134，按严格 64env 完整首回合进行对应评估。这个既有失败 seed
是有意的压力测试，结果只能标 Probe，不能用于普遍 claim。

预定推进门：新 arm 在两次评估合计上比既有 effect-rank 高至少 10pp，
且比 off 高至少 5pp；否则标 `UNPROMISING` 或 `UNCLEAR`，不扩大相同方法。

## Budget and stop

1 GPU（物理 5）；预计 <30min，新增 <2GB。启动前检查 GPU；代码提交、
源 checkpoint、CmLite 和运动输入 SHA 固定；输出目录必须新建。
输入/源码漂移、GPU 冲突、非有限值或训练/评估合同失败则停止并记录。

## Result

待运行。
