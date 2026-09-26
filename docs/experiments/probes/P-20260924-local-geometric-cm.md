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

Status: UNPROMISING（动作条件信息未过门）

Key evidence: `agent_local_geometric_cm_300` COMPLETED；CPU-only；
六区域 12,932 参数、物体/手各 256 点。训练 768、测试 192 条。
固定 300 更新步，运动接触 64 条测试：Cm EPE 5.75mm，
零运动 11.61mm；但同架构零手流 6.04mm，仅改善 4.8%，
动作置乱 5.93mm，仅恶化 3.1%，均远未过预设 15% 动作信息门。
静止接触 Cm EPE 2.86mm，零运动 0.41mm，存在明显假运动。
这说明当前配方在已执行动作的观测数据上主要学习运动先验，
不能据此认为学到了可供策略选择动作的因果效应。

只读检查旧“first-grip counterfactual”两臂文件：seed97 仅
1/64 条预干预状态匹配，seed98 0/64；replay seed98 也 0/64。
因此旧文件不能提供同状态不同动作的有效物理监督/验证。

## Decision update

不把当前小 Cm 接入 PPO，不继续只调网络宽度或 loss。
下一步应先获得**同状态、不同动作、都在仿真中执行**的配对转移，
并严格验证干预前状态一致和同动作重复的仿真一致性；在这种数据上
测试 Cm 是否能区分行动效应。若无法构建可靠配对，才考虑直接
`s,a→s'` 的其他结构/监督。该判断不否定几何 Cm 本身。

## Artifacts

`src/task/CmResidual/tools/probe_local_geometric_cm.py`
`outputs/CmResidual/agent_local_geometric_cm_300/report.json`
`outputs/CmResidual/agent_local_geometric_cm_300/run_manifest.json`
