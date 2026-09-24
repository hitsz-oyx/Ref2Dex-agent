# P-20260924-randomized-geometric-cm

date: 2026-09-24
branch: agent/cm-randomized-action-effect
classification: Decision

## Question

随机真实动作干预提供了足够的动作覆盖后，一个小型局部几何 Cm
能否在未见 seed 上预测**动作条件**的一步物体效应，超过仅看状态
与零物体运动的对照？它与普通 raw state+action MLP 谁更有效？

## Hypothesis and decision

H1: 由混合动作幅度数据校准的在线手流 + 六区域近物几何 Cm，
在未见 seed146 ±0.3、seed148 ±0.1 的两组实际执行接触转移上，
平均平移 EPE 均比同结构零手流模型低 ≥20%，并且模型对
“把相同当前输入动作改成 +δ/-δ”的平均预测 z 效应与随机试验的
真实加减臂平均差相差 ≤25%。

通过：进入小型在线候选动作评分 Probe，仍需 matched Cm-on/off
才能形成策略效用结论。若 raw state+action MLP 同样好或更好，
应简化/改造几何 Cm 后再投入在线策略。

不通过：不再把当前几何 Cm 接 PPO；检查几何 token 是否保留
动作方向、损失是否被物体状态先验主导，或转向更直接的局部
接触效应表示。随机干预效应本身不因此消失。

## Minimal protocol

训练 seed145 ±0.3 + seed147 ±0.1 的 1188 条接触实际转移；
测试未见 seed146 ±0.3 + seed148 ±0.1。所有动作确实在 DExplore
仿真执行，不用官方 actor。当前 `q,dof_vel,action,object_state` 经过
已校准 hand-flow 和手/物几何，形成六区域 token；未来 `next_q`
仅作 actuator 的离线训练标签，不进入 Cm 的在线特征。固定 400
更新步、相同 minibatch 日程，比较几何 Cm、同架构零手流、
raw state+action MLP，以及零物体运动。

模型对两个候选动作的输出差仅是**预测的**条件效应；真实数据
只有随机化平均效应，不能把每个模型候选输出当作逐样本真值。

## Budget and stop

CPU 2 threads、GPU 0；wall <=30 min；输出 <10MB。
输入 SHA 漂移、训练/测试 seed 混淆、非有限 loss 或预算超限停止。

## Result

Status: UNCLEAR (EPE/action controls pass; small-dose ATE calibration misses gate)

Evidence: CPU-only `agent_randomized_geometric_cm_400` COMPLETED in 9s.
1188 training randomized contact transitions, 12,932 geometric parameters.
Held-out seed146 ±0.3: geometric EPE 11.25mm, same-architecture zero-flow
16.79mm, raw state+action MLP 13.86mm, zero-motion 16.63mm; model average
intervention effect +20.40mm vs randomized observed +26.00mm (21.5%
underprediction). Held-out seed148 ±0.1: geometric EPE 5.81mm,
zero-flow 7.66mm, raw MLP 6.53mm, zero-motion 7.67mm; model effect
+6.83mm vs observed +10.42mm (34.4% underprediction). Geometric Cm
beats zero-flow by 33%/24% and raw MLP by 19%/11%, but H1's <=25%
effect-calibration gate fails at small dose. The model effect is positive
on 87%/70% of test states, not uniformly.

## Decision update

Do not yet connect this Cm to PPO. The next cheap decision Probe should
test whether model-predicted *relative* effects stratify the actual
randomized treatment response on untouched test seeds. If ranking is
uninformative despite better EPE, the current Cm cannot guide actions.

## Artifacts

`src/task/CmResidual/tools/probe_randomized_geometric_cm.py`
`outputs/CmResidual/agent_randomized_geometric_cm_400/report.json`
