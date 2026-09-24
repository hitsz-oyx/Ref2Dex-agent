# P-20260924-cm-online-action-boost

date: 2026-09-24
branch: agent/cm-randomized-action-effect
classification: Decision

## Question

已学到随机处理效应异质性排序的几何 Cm，在真实 DExplore
闭环中选择接触时的小幅腕部上抬，是否比不干预的自训练 actor
以及“接触就上抬”的简单规则更稳定抓取？

## Hypothesis and decision

H1: 固定自训练 e260 actor 和 s3 轨迹，Cm-on 在新 seed149/150
的首个完整 episode 上提升标准 held-lift 成功率，且超过
always-boost 简单规则。只有两 seed 均不负、合计相对两个对照
各 >=8pp，才考虑更大样本/多训练 seed；任一门失败则
`UNCLEAR/UNPROMISING`，不把离线 CATE 结果当策略效用。

工程 smoke 先在 seed149 各 16 环境运行 base、always、Cm；
若接线/延迟/有限性失败，立即停止，不升级 64 环境 Probe。

## Minimal protocol

三个 arm 共享 actor checkpoint、motion 数据、评估脚本和 seed；
动作只在全局控制步 50–150 的偶数步、当前实际手物接触时可改。
`base` 原 actor 动作；`always` 接触就腕部 z 命令 +0.1；
`Cm` 每个接触状态用固定几何 Cm 预测腕部 z 的 +0.1/-0.1
两个候选的一步物体 z 位移差，仅当该预测差 >12.06mm 才
对原 actor 命令 +0.1。12.06mm 是此前未见 seed148 的固定
最高四分位阈值，不从本次 seed 调整。Cm 模型推理不使用未来
状态，也不使用官方 actor checkpoint。

对照 always 能区分“Cm 选择有信息”与“固定上抬就有用”。
独立 GPU PhysX 环境不能作为逐样本配对，结果只作小 Probe，
正式因果 policy utility 仍需多训练 seed matched Validation。

## Budget and stop

每次 1 张空闲 GPU、≤64 环境；单 run <=20 min，六次
Probe 总计 <=60 min，产物 <100MB。模型/输入漂移、GPU
冲突、非有限预测、闭环异常或超预算停止。

## Result

Status: PENDING

Evidence: pending

## Decision update

pending
