# P-20260924-local-geometric-cm

date: 2026-09-24
branch: agent/cm-executed-handflow

## Question

在校准一步实际手运动后，一个低延迟、显式局部手—物几何的 Cm，
能否从自训练策略的仿真真实执行转移学到超过零运动/零手流的
动作条件物体效应？

## Hypothesis

H1: 以 6 个手区域（掌部+五指）的近物几何/校准手流构造 Cm tokens，
在未训练 seed95/96 的接触且物体移动样本上，局部平移 EPE 比零物体
运动和同架构零手流模型均低至少 15%，且动作置乱使误差变差。

Alternative: 当前已执行转移只支持物体惯性/运动先验，几何 Cm
没有动作特异信息；需改数据采集为同状态多动作物理反事实或换监督目标。

## Decision

H1 通过：再做新 seed、同状态物理动作反事实与在线延迟 Probe，
然后才考虑接 PPO。

若不通过：不继续堆 Cm→PPO 接法，先审计数据的动作覆盖/接触边界，
必要时收集同状态多动作转移；保留跨手混合预训练为后续研究。

## Minimal protocol

固定 Inspire 采样几何：物体 256 点、手 256 点；只读使用已执行
DExplore 转移。每条样本由当前 `q,a,q_prev,object_state` 得到
预测实际手流；从同一当前状态计算六区域的最近物体距离、方向、
局部接触位置、法线和校准手流统计，形成小型 Cm tokens。
物体下一状态只作监督/分层。对照：同结构同更新步数零手流；
静止物体基线。评估新 seed95/96 的运动接触、静止接触、非接触，
并做动作流置乱。先工程 smoke，再小 Probe，不做 PPO。

## Budget

CPU 2 threads、GPU 0（若出现真正空闲卡可择一使用）；
Probe wall <= 60 min，输出 < 1 GB。

## Stop condition

输入/模型 SHA 漂移、非有限值、训练/测试泄漏、输出超预算、
GPU 被他人占用或 wall 超预算即停止。

## Result

Status: PENDING

Key evidence: pending

## Decision update

pending

## Artifacts

`src/task/CmResidual/tools/probe_local_geometric_cm.py`
