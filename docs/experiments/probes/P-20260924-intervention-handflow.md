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

Status: PENDING

Evidence: pending

## Decision update

pending

## Artifacts

`src/task/CmResidual/tools/probe_intervention_handflow.py`
