# Cm PPO 权重分量离线 Probe

probe_id: `PROBE-20260923-CM-WEIGHT-COMPONENTS`

date: 2026-09-23

branch: `agent/cm-weight-mechanism`

## Question

V1.52 证明对齐权重优于置乱权重，但 `p_contact * clipped_abs_dz` 的哪一部分
更可能携带一步物理效应信息？应优先做哪个在线分量消融？

## Hypothesis and decision

若两部分各有独立的真实效应关联与动作敏感性，下一步做小规模 PPO
contact-only / effect-only 消融。若只有一部分有明显信号，先测该单分量
与 joint 的在线对照；若两部分都没有，暂停当前权重接法、改查状态混杂或模型目标。

## Minimal protocol

使用已有自训练 s3 e260 策略 seed95/96 的完整首 episode 非终止真实转移，
冻结 V1.37 CmLite。对接触概率 `p`、`clip(|pred_dz|/3mm,0,1)`、二者乘积，
计算与“真实接触且一步 |dz|≥3mm”事件的排序 AUC、真实 |dz| 相关系数、
top/bottom decile 事件率；在相同状态打乱动作作诊断负对照。
这里只能判断下一步优先级，不能当作在线 PPO 因果证据。

## Budget and stop condition

CPU、无新模拟、预计数分钟、结果 <1 MB。若输入或 checkpoint SHA 不符、
首 episode 过滤失败、任一预测非有限，立即停止。

## Result

Status: PROMISING

Key evidence: seed95/96 首 episode 非终止转移 33,216/33,043 条；
“真实接触且 |dz|≥3mm”事件率 16.7%/23.3%。接触分量 AUC
0.765/0.751，位移分量 0.888/0.873，联合 0.905/0.892；同状态打乱动作后
联合降为 0.853/0.838。动作打乱后的联合权重平均绝对变化 0.120/0.143。
结果文件：`outputs/CmResidual/agent_cm_weight_component_probe/probe.json`。

这说明两分量均有排序信息，位移分量更强，联合优于单分量；但动作打乱后
仍保留很高 AUC，表明大量信息可能来自状态。此 Probe 不证明 PPO 因果效用。

## Decision update

进入极小的在线分量消融：保持每个 PPO step 的活跃样本联合权重多重集
精确相同，只改变“由哪一个 Cm 分量决定谁得到大权重”。比较 contact-rank、
effect-rank 与既有 joint/off 的短期抓取。若结果不清晰，不扩大成多 seed Validation。
