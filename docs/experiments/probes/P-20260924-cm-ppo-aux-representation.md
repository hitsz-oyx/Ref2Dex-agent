# P-20260924-cm-ppo-aux-representation

date: 2026-09-24
branch: agent/cm-ppo-aux-representation
classification: Decision

## Question

冻结真实仿真执行转移训练的 Cm，通过训练期辅助预测目标塑造
PPO actor 的共享表示，是否比完全相同的 Cm-off 微调获得更好
的未见 seed held-lift？

## Hypothesis and decision

自训练 e260 actor 为共同起点，训练 seed75/76，两臂同网络、
同初始权重、同 PPO reward/curriculum/优化设置。
`on` 在 actor 最后一层共享特征上加三维线性辅助头，预测
冻结 raw-action Cm 对 wrist-z ±0.1 候选的五步接触保持、
一步物体高度和五步物体高度差；`off` 建同样辅助头、同样
计算 teacher，但 loss 系数为 0。辅助目标只在当前真实接触
状态监督，不改 reward、advantage、actor 动作或最终推理网络。
系数预先固定 0.002，不从测试 seed 调节。

先做 on/off 各 2 个 epoch 工程 smoke。若双方从固定 e260
严格恢复、teacher/辅助 head 有限、on 辅助 head 与 actor trunk
确有梯度而 off 没有、最终 checkpoint 能用原评估器加载，
才进行两训练 seed 的 e260→e300 最小 matched Probe。
新评估 seed159/160 各 64 env；主指标为首完整 episode 的
held-lift 成功率。只有 on 相对 off 合计 ≥8pp，且每个训练
seed 合计不负，才升级多训练 seed/置乱 target Validation。
第一训练 seed 若严重负向（两个评估 seed 合计 ≤−8pp），
提前停止第二训练 seed，不在结果上调系数。

## Minimal implementation

冻结 Cm checkpoint 只在 rollout 生成辅助监督；目标按固定
尺度归一并裁剪。辅助 head 不写进 actor 网络 checkpoint，
最终 checkpoint 的 actor/model state dict 与原 DExplore 完全
兼容；辅助 head 参数与 optimizer state 另外记录。评估不加载
Cm。两臂都使用同一代码路径，只有预设 loss 系数不同。

## Budget and stop

每次训练 1 张空闲 GPU、64 env、e300 截止，预计单 run
<20min；最多两张 GPU 并发，但优先顺序运行。每次评估
1 张空闲 GPU、64 env；整个 Probe <60min，新增产物 <5GB。
输入/代码 SHA 漂移、恢复不精确、on/off 初始 actor 不同、
非有限预测/梯度、GPU 冲突、超过预算即停止。

## Result

Status: PENDING

Evidence: pending

## Decision update

pending
