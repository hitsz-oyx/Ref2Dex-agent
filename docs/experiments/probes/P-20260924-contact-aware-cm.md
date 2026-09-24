# P-20260924-contact-aware-cm

date: 2026-09-24
branch: agent/cm-randomized-action-effect
classification: Decision

## Question

六区域几何 Cm 能否从执行前状态与候选动作同时预测一步物体
效应和随后五步的接触保持，并在未见随机化 seed 上识别动作
对接触的异质影响？这决定是否值得再做在线 Cm 候选评分。

## Hypothesis and decision

用 seed151/152 的随机 ±0.1 真实执行随访训练，seed153/154
只测试。几何 Cm、相同特征但动作手流置零的 state-only 对照、
小型 raw state+action MLP 共享批次与更新预算。

若几何 Cm 在两个未见 seed 上均能预测接触处理效应为负，且
预测接触效应最高−最低四分位的真实 RCT 接触效应差合并 ≥5pp、
environment-cluster 95% CI 不含 0，并且至少不劣于 raw MLP，
则设计一次安全的在线 matched Cm-on/off Probe；否则不做
接触排序在线搜索，转向训练期 representation/auxiliary route。
若处理效应几乎均匀，即使平均方向正确也不够形成 action ranking。

## Minimal protocol

每种模型三头：一步物体局部位移、五步物体局部位移、五步手物
接触比例。线上可用输入仅为当前 `q,dof_vel,object_state,action`
与固定运动学/执行模型；不使用未来 `next_q`。训练不接触测试
seed、测试标签不用于调结构或阈值。

比较标准预测误差与随机处理效应排序。分组由模型在处理前
为 ±0.1 两个候选的接触预测差决定，真实分组效果由随机分配
估计；这不是逐状态物理反事实。

## Budget and stop

采集各 1 张空闲 GPU、64 env、单 run <30min；训练/分析 CPU
2 threads、≤1000 更新、总 <60min，新增产物 <200MB。
输入漂移、非有限训练、接触标签损坏或预算超限停止。

## Result

Status: PENDING

Evidence: pending

## Decision update

pending
