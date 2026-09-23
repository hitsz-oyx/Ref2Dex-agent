# P-20260923-mixed-cm-sim-transfer

date: 2026-09-23
branch: agent/cm-mixed-pretrain-sim-adapt

## Question

已混合 GRAB MANO + 几何重定向 Inspire 预训练的 ObjectInteractionCm V1.3，
能否在自训练策略的仿真真实执行转移上提供可迁移的一步物体效应信号？

## Hypothesis

H1: 冻结模型用在线可得的 nominal action hand flow，至少在接触样本上
优于零物体运动，并对动作置乱有方向敏感性。

Alternative: 预训练的运动学效应与真实物理转移/动作执行语义不匹配；
需要 sim-transition adaptation，不应直接接 PPO。

## Decision

H1 若成立：测试少量真实仿真转移微调，比较预训练与从零训练；不直接宣称策略效用。

若不成立：仍可做一次小规模 matched finetune-vs-scratch Probe，
但先校正 hand-flow 执行失配，禁止直接以冻结模型接入 PPO。

## Minimal protocol

固定自训练 PPO 的 seed95/96 首 episode 转移；每 seed 接触/非接触各 8 条，
用同一当前状态构造 nominal action flow、动作置乱 flow、仅离线诊断用的
oracle next-q flow。报告 rigid object point-flow EPE、零运动 EPE、有效交互比例、
动作输入敏感性。模型和转移按 SHA 固定。此 Probe 仅区分输入合同/迁移差距，
并不声称随机动作置乱是真实物理反事实。

## Budget

GPU: 最终采用 CPU（其他 GPU 被占用）；wall time: < 10 min；storage: < 1 MB。

## Stop condition

GPU 被他人占用、非有限输出、geometry/coordinate-frame 合同不匹配立即停止。

## Result

Status: UNPROMISING（冻结直接迁移）

Key evidence: CPU 运行每个 seed 接触/非接触各 8 条。接触样本：

| seed | nominal EPE | oracle next-q EPE | zero EPE | nominal-oracle hand RMS |
| --- | ---: | ---: | ---: | ---: |
| 95 | 38.41 mm | 6.85 mm | 6.50 mm | 44.55 mm |
| 96 | 33.63 mm | 2.35 mm | 11.70 mm | 44.28 mm |

nominal 在两个 seed 都远逊零运动；oracle 只用于诊断，不能在线使用。
随机动作置乱并未稳定变差（seed95 30.57mm、seed96 47.65mm），
故无可靠动作排序结论。非接触样本接近静止，但模型在交互无效时
仍输出约 24–26mm 运动，部署时须显式 `sample_valid` 门控。
样本非常小，仅作为迁移/输入合同 Probe，不能推广到总体误差。

## Decision update

放弃冻结模型直接接 PPO。下一步构造同一批仿真真实执行转移的
预训练初始化 vs 从零训练的小规模 matched 微调对照；模型输入保持
在线可得的当前状态 + 动作/nominal hand flow，目标用执行后的真实转移。
穿模/运动学标签与仿真标签分开；预训练资料的 `hand_min_object_distance_m`
是无符号距离，不能当作穿模深度。若要过滤严重穿模，需另算 signed SDF。

## Artifacts

`src/task/CmResidual/tools/probe_mixed_cm_sim_transfer.py`
`outputs/CmResidual/agent_mixed_cm_sim_transfer_cpu_probe.json`
