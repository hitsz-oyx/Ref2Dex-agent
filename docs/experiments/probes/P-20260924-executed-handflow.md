# P-20260924-executed-handflow

date: 2026-09-24
branch: agent/cm-executed-handflow

## Question

只用决策时可获得的当前关节 `q_t`、当前动作 `a_t` 和上一控制步
`q_{t-1}`，能否预测 DExplore 的实际下一步手运动，替代错误的
“PD 目标瞬时到位” hand flow？

## Hypothesis

H1: 简单的逐关节执行模型 `Δq_actual = α Δq_target + β Δq_previous`
在未见的 e260 seed95/96 接触状态上，把手表面一步位置 EPE 降至
静止手 baseline 的 50% 以下，并且不比 oracle flow 泄漏未来输入。

Alternative: 当前状态/上一状态/动作不足以表示接触下执行动态，
需要记录真实预步速度、接触信息或改为直接 `s,a→物体效应` Cm。

## Decision

若通过：用冻结的因果执行模型提供 hand flow，再训更小的局部几何 Cm；
同时对照用原 nominal flow 和真实事后 hand flow 的离线 oracle。

若失败：先测小型非线性模型；如果仍失败，不投入更大动作动力学路线，
改成 Cm 直接接受 `q,a`，或补录预步速度。

## Minimal protocol

训练：自训练 DExplore e160 seed74、e180 seed78 的完整首 episode
非终止转移。测试：e260 seed95/96，固定接触/非接触分层各 64 条。
排除 episode 初始步，避免上一关节位置跨 reset。比较静止手、
PD 目标、全局 gain、逐关节 gain、逐关节 gain+momentum。
主指标为同一 Inspire 几何采样点的一步位置 EPE（mm），
另外报告 actual hand motion 与预测动作变化；测试 `next_q` 只作标签。
样本为已执行动作的观测数据，不是同状态的物理反事实。

## Budget

CPU 2 threads、GPU 0、wall <= 15 min、输出 < 2 MB。
先 smoke，再固定完整 Probe。

## Stop condition

输入 SHA 漂移、非有限输出、episode 对齐失败、wall 超预算即停止。

## Result

Status: PROMISING（已执行动作的一步手运动预测）

Key evidence: CPU-only run `agent_executed_handflow_64` COMPLETED；训练转移
e160 seed74 + e180 seed78 的完整首 episode，测试 e260 seed95/96
各接触/非接触 64 条，严格排除 episode 初始步。接触手表面 EPE：

| seed | 静止手 | PD 目标瞬时到位 | 逐关节 gain | gain + 上一步动量 |
| --- | ---: | ---: | ---: | ---: |
| 95 | 10.46 mm | 45.93 mm | 5.83 mm | 2.70 mm |
| 96 | 8.26 mm | 43.43 mm | 5.48 mm | 2.88 mm |

最后一列分别为静止基线的 25.8%/34.8%，过预设 50% 门。
非接触也改善到 4.10/4.61mm，静止基线 15.00/19.41mm。
模型由当前 `q,a`、上一步 `q_prev` 构造，测试 `next_q`
只作标签。打乱动作后的预测手表面位置平均变化 7.46/8.99mm，
但这不是已执行的物理反事实。该 Probe 不能证明物体效应或策略效用。

## Decision update

保留简单逐关节 gain+momentum 作在线因果手运动输入；不启动大规模
执行动力学网络。先用冻结混合 Cm 检查校准手流是否修复物体效应，
若仍不行，再训练更小的局部几何 Cm。后续正式评估需新 seed。

## Artifacts

`src/task/CmResidual/tools/probe_executed_handflow.py`
`outputs/CmResidual/agent_executed_handflow_64/report.json`
`outputs/CmResidual/agent_executed_handflow_64/run_manifest.json`
