# P-20260924-randomized-action-effect

date: 2026-09-24
branch: agent/cm-randomized-action-effect
classification: Decision

## Question

在无法得到可信的逐样本仿真反事实后，对接触中的真实仿真动作做
随机 ±z 干预，是否能识别可供动作条件 Cm 学习的平均一步物体效应？

## Hypothesis and decision

H1: 自训练 s3 策略的接触状态中，腕部 z 命令随机 ±0.3 后实际手 z
运动产生明显差异，且物体一步竖直位移的加/减臂差异 >=0.25mm、
分步分层置换 p<=0.1。至少收集 100 条接触干预，实际手 z 差异
目标 >=5mm。

H1 通过：收集新 seed 的小量随机干预转移，训练动作条件局部 Cm
与同结构零动作对照，并测试模型能否预测干预效应；过门后再考虑 PPO。

H1 不通过：不投入该 z 方向 Cm 训练；检查实际动作传递/接触几何，
再决定是否改用手指闭合或其他局部接触方向，而不是宣称 Cm 无效。

## Minimal protocol

固定自训练 e260 actor 与 s3 Inspire 轨迹；原 DExplore config、不使用
官方 actor。headless 64 环境，seed145，在全局步 50,60,...,150 的
预接触（当前手与物体都存在 >0.1 力）环境中，每步随机均衡分配
腕部 z 命令 +0.3/-0.3，其他环境执行原 actor 动作。每个分配动作
都在仿真中真实执行；记录干预前状态、原/实际动作、下一步状态。
分配独立于干预前状态，但连续干预会影响后续状态；因此只能估计
当前随机化状态分布上的平均处理效应，不能称为同状态逐样本反事实。
先做一个短 Probe，不运行 PPO。

## Budget and stop

一张真正空闲 GPU；wall <=30 min；输出 <100MB。
输入 SHA 漂移、GPU 被占用、非有限状态或超预算立即停止。

## Result

Status: PENDING

Evidence: pending

## Decision update

pending

## Artifacts

`third_party/DExplore/dexplore/evaluate_randomized_action.py`
