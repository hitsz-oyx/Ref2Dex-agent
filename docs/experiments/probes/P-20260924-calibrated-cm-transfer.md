# P-20260924-calibrated-cm-transfer

date: 2026-09-24
branch: agent/cm-executed-handflow

## Question

把已经通过小 Probe 的因果执行手流校准接到冻结混合 MANO/Inspire
ObjectInteractionCm V1.3，能否恢复仿真真实执行的一步物体效应预测？

## Hypothesis

H1: 校准 hand flow 在未见自训练 PPO seed95/96 的接触样本上，
使冻结 Cm 的物体点流 EPE 比零运动基线至少低 20%，且明显优于
PD 目标瞬时到位的 nominal flow。

Alternative: 输入修复仅解决部分域差；混合预训练的运动学物体效应
无法直接迁移到仿真物理，需重训/重设计更小局部 Cm。

## Decision

H1 若成立：测动作反事实排序和延迟，再决定是否蒸馏/微调小模型。

若不成立：不直接使用冻结混合 Cm；以校准执行手流为输入训练
小型局部几何 Cm，并以真实仿真转移监督。

## Minimal protocol

冻结同一 Cm 权重与同一仿真状态，仅替换 hand-flow 输入：
PD target、校准执行预测、事后 `next_q` oracle、零流。测试 e260
seed95/96，接触/非接触各 16 条；目标为真实执行后物体刚体点流。
主指标接触物体点流 EPE；oracle 只用于误差分解，绝不进入在线模型。
已执行动作样本不构成物理反事实。

## Budget

CPU 2 threads、GPU 0、wall <= 10 min、输出 < 2 MB。

## Stop condition

任一权重/输入 SHA 漂移、非有限值或 wall 超预算即停止。

## Result

Status: UNCLEAR（输入修复有效，冻结物体效应未可靠过门）

Key evidence: `agent_calibrated_cm_transfer_64` 在 seed95/96 各取
64 条接触、64 条非接触转移；CPU-only，冻结同一模型和同一状态。

| seed 接触 EPE | nominal | 校准手流 | 事后 oracle | 零手流 | 零物体运动 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 95 | 41.02 mm | 5.97 mm | 5.29 mm | 8.65 mm | 5.87 mm |
| 96 | 35.25 mm | 6.21 mm | 4.92 mm | 10.95 mm | 9.14 mm |

校准输入大幅改善两个 seed，并接近事后 oracle；但是 seed95 仍略差于
零物体运动，预设“两 seed 稳定优于零运动 ≥20%”的实质门未通过。
这不是 Cm policy utility 的证据。样本仍来自单轨迹，seed95/96
此前已用于诊断；此处只作机制 Probe，不作独立 Validation。

## Decision update

不以冻结 V1.3 物体效应头直接接 PPO。保留因果校准手流，
训练更小的局部几何 Cm，在真实仿真执行转移上与同架构零手流对照；
若小模型仍无法越过零运动并识别动作对应关系，暂停新的 PPO 接法。

## Artifacts

`src/task/CmResidual/tools/probe_calibrated_cm_transfer.py`
`outputs/CmResidual/agent_calibrated_cm_transfer_64/report.json`
`outputs/CmResidual/agent_calibrated_cm_transfer_64/run_manifest.json`
