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

Status: UNPROMISING（当前冻结 Cm + 接触窗口 +0.1 抬腕规则）

工程 smoke：seed149 ×16 的 base、always、Cm 均正常结束，held-lift
均为 10/16；只验证接线，未据此筛选阈值。

预先固定协议后，在新 seed149/150 ×64 的首 episode 上运行三臂，
六个 run 均 `COMPLETED`：

| seed | base | always +0.1 | Cm 选择 +0.1 |
| --- | ---: | ---: | ---: |
| 149 | 41/64 (64.06%) | 40/64 (62.50%) | 42/64 (65.62%) |
| 150 | 46/64 (71.88%) | 33/64 (51.56%) | 44/64 (68.75%) |
| 合计 | 87/128 (67.97%) | 73/128 (57.03%) | 86/128 (67.19%) |

Cm 相对 base 为 −0.78pp，且 seed150 单独为 −3.13pp；相对
always 为 +10.16pp。Cm 选择比盲目上抬少伤害，但没有提高
baseline 成功率，未过预设“两 seed 均不负且合计各 +8pp”门槛。
两个 seed 的平均 reward 和手物接触比例也都低于 base：
seed149 reward 184.83→171.61、接触比例 .57288→.54980；
seed150 reward 199.30→174.56、接触比例 .62086→.56383。
Cm 分别从 2905/2873 个接触 eligible 状态中选择 661/671 次
上抬，每个 episode 评估 51 个候选时刻；不是 Cm 未被调用。

产物：
`outputs/CmResidual/agent_cm_online_probe_s{149,150}_n64_{base,always,cm}/`
下的 `run_manifest.json`、`results.json`、`selector.json`。

## Decision update

停止这条“单步物体 z 效应大就直接上抬”的在线接法，不在本次
seed 上调阈值或窗口。离线 CATE 排序成立与策略效用是不同命题；
当前数据支持局部效应信息，却不支持其通过该规则改善长期抓取。
下一步应改为能考虑接触保持/长期结果的 Cm 用法，并设置新的
matched Cm-on/off Probe；不得把本卡写成“Cm 无用”的总体结论。
