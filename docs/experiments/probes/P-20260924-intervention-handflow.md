# P-20260924-intervention-handflow

date: 2026-09-24
branch: agent/cm-randomized-action-effect
classification: Decision

## Question

在随机真实动作干预的较宽动作范围，原先用观测策略动作校准的
`action→actual hand motion` 是否还能给 Cm 提供准确的在线手流？
能否用实际仿真转移重新拟合一个低延迟、可在线使用的模型？

## Hypothesis and decision

H1: 仅用当前 `q,dof_vel,action` 拟合的逐关节 action+velocity
线性模型，在未见 seed146 的 ±0.3 与 seed147 的 ±0.1 接触干预上，
手表面 EPE 比旧观测动作 gain 模型至少降低 30%，且腕部 z 随机
加减臂的预测运动差距离真实差异不超过 20%。

通过：将该校准作为后续轻量 Cm 的在线手流输入，再检验物体效应。
不通过：先检查动作饱和/速度与接触状态的非线性，尝试小型非线性
actuator；不直接把未来 `next_q` oracle 输入 Cm。

## Minimal protocol

训练只用 seed145 ±0.3 的实际执行接触转移（560 条）；测试
seed146 ±0.3 和 seed147 ±0.1。比较静止手、PD 目标瞬时到位、
速度外推、旧观测 gain、新 action-only、新 velocity-only、
新 action+velocity；模型仅使用当前在线可得量。CPU-only，
闭式逐关节拟合，无 PPO。

## Budget and stop

CPU 2 threads、GPU 0；wall <=20 min；输出 <5MB。
输入 SHA 漂移、非有限值或预算超限停止。

## Result

Status: UNCLEAR (strict H1 gate fails; mixed-dose follow-up is useful)

Evidence: CPU-only `agent_intervention_handflow_s145_train_s146147_test`
COMPLETED. On held-out seed146 ±0.3, new action+velocity hand surface EPE
11.81mm vs old observational gain 20.37mm (42% lower); predicted wrist-z
plus/minus motion 74.52mm vs actual 75.89mm. On held-out seed147 ±0.1,
new EPE 10.30mm vs old 8.66mm (worse), although predicted wrist-z contrast
24.98mm vs actual 27.34mm, while old predicts only 16.89mm. Velocity-only
EPE 34.36/13.88mm, so action input carries substantial information.
Strict H1 EPE gate across both magnitudes fails. A cheap mixed-dose fit
using seeds145 ±0.3 and147 ±0.1 will be tested on untouched seeds146 ±0.3
and148 ±0.1 before deciding whether the linear actuator is sufficient.

Mixed-dose follow-up `agent_intervention_handflow_mix_s145147_train_s146148_test`
COMPLETED on 1188 training interventions. Held-out seed146 ±0.3:
new action+velocity EPE 12.21mm vs old 20.37mm; predicted wrist-z
plus/minus 74.76mm vs actual 75.89mm. Held-out seed148 ±0.1:
new EPE 7.49mm vs old 8.49mm; predicted contrast 25.04mm vs actual
27.45mm. Thus it beats the old gain model at both doses and captures
intervention response within ~9%, but its small-dose EPE gain is only
11.8%, short of the original ≥30% H1 threshold.

## Decision update

Use the mixed-dose model as a provisional online-only Cm input for a cheap
model Probe because it improves both held-out magnitudes and accurately
tracks the randomized action response. Do not call the actuator solved or
claim policy utility. Keep future `next_q` out of online inputs.

## Artifacts

`src/task/CmResidual/tools/probe_intervention_handflow.py`
`outputs/CmResidual/agent_intervention_handflow_mix_s145147_train_s146148_test/report.json`
