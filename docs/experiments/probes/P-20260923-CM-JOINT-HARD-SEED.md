# Probe: 早期 joint Cm actor 权重能否通过失败 seed 的压力测试？

probe_id: `P-20260923-CM-JOINT-HARD-SEED`

date: 2026-09-23

branch: `agent/cm-postvalidation-diagnosis`

status: PLANNED

## Question and decision

正式验证否定 effect-only 排序的稳定效用；有向正效应排序的 seed74
Probe 也失败。早期 V1.51 的正结果实际使用接触概率×绝对效应的
**原始 joint 权重**，并未跨训练 seed 验证。本 Probe 用已失败的
训练 seed74 给它一次最低成本压力测试，以决定是否还值得对这个
具体 joint 接法做正式多 seed 验证。

若 joint 在该 seed 上同时高于 effect-rank 和 Cm-off 至少 8pp，
再考虑独立训练 seed 的 Probe/Validation；否则不继续调整 actor
样本加权的局部分数，转向 Cm 参与 critic/表征等不同机制。

## Minimal protocol

从相同自训练 DExplore s3 e260 checkpoint、相同 CmLite V1.37、
相同 corrected_r2 运动输入、相同训练 seed74 和 e300 预算出发。
只用已有 `joint` 模式，保持真实奖励、PPO 超参数、课程和网络一致。
不使用官方 actor。严格评估 seed133/134、每次 64env 完整首回合；
与此前固定的同训练 seed74 effect-rank/off 结果比较。

这两个评估 seed 的对照结果已知，且 seed74 是事后挑选的困难 seed，
故结果只用于停损/继续决策，不能当成独立确认或论文 claim。

## Budget and stop

1 GPU，预计 <30min、新产物 <2GB。代码 commit、输入 SHA、GPU
和唯一 run_id 固定后启动。训练或评估合同失败、GPU 冲突、输入漂移
则停止，不更改 seed 或门槛。

## Result

待运行。
