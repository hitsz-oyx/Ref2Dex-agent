---
schema: ref2dex.probe.v2
probe_id: P-20261016-wam-learned-short-goal
experiment_id: P-20261016-wam-learned-short-goal
date: 2026-10-16
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-wam-short-goal
probe_index_in_family: 1
seed_pool: probe
seeds: [246, 247]
decision_changed_if_positive: test native realization of a predicted short goal using WAM versus geometric interaction priors
decision_changed_if_negative: separate current-behavior prediction from goal selection and diagnose uncertainty before native rollout
status: UNCLEAR
run_id: learned-short-goal-20261016-r1
---

# 从当前观测预测短期终点，检验可部署的WAM条件桥接

Class: Decision。承接任务读出Probe，服务物理预训练帮助小机器人数据PPO。
核心问题：不读取未来时，能否从当前/过去观测预测0.8s物体终点，并保留WAM手物
预测的优势？这里先预测旧混合行为的实际终点，不称任务最优目标或已认证可实现目标。

## Decision Note

先前同oracle条件下WAM手部误差比几何插值低约50%；但整任务goal与短期终点语义
不同，scheduled条件未胜zero。最低成本是重用已提取TRAIN状态、预测实际48步SE(3)，
再以模型预测终点运行冻结WAM；不需新BC、动作标签、在线采集或预训练。
若可部署条件满足下面gate，才研究原生执行；若不满足，不用oracle冒充在线结果，
先区分终点误差、当前行为多义性及学习泛化，不加大PPO预算。

## 固定协议

- 原始行为：`shadow-value-scale1-32obj-20261016-r1/rank00` first/last episode；
  输入缓存：`task-readout-20261016-r1/features/features.pt`，WAM final80k保持冻结。
- 继承24/8整对象TRAIN内部拆分、seed246/247。7ticks `[6,28,51,74,97,120,143]`，
  1792状态、1344fit/448held；同对象所有env/episode不跨组。此面板已被探索使用，非独立TEST。
- 两臂raw-query、wam-mean：只读取原current300+known goal3+remaining1，
  raw臂读取全部observed CAD/手/历史/goal tokens；WAM臂读取已缓存task-goal条件三类型
  mean。无oracle/未来/当前待执行动作输入，无教师动作监督，原生过去动作仍属状态。
- 小型128D读出→9D刚体终点（translation3 + 6D两列旋转、正交化），输出初始identity。
  对32个当前CAD query点施加该刚体变换，确保goal点不是非刚性任意回归。
- 实际t+48物体位姿仅作为训练标签/评估真值；loss为点位移/0.05m后的SmoothL1(beta1)。
  输入统计只用fit对象；seed246、batch seed247、AdamW3e-4/WD1e-4、batch128、
  gradient clip1、固定1000步，不按held挑checkpoint。保存两个固定最终模型。
- 同面板报告endpoint48点误差、translation误差，原始zero/scheduled/task条件作基线。
  分fit/held，另在实际48步到任务goal距离减少>2cm的progress子集报告；不按未来筛选训练。
- 将每个最终模型预测的endpoint32点传入冻结WAM；报告hand/scene16/32步误差，
  与已有同面板WAM-zero/scheduled及同预测终点的几何插值比较。actual future只用于打分。
- Gate预先固定：held endpoint点误差比zero和scheduled中更优者低>=10%；
  WAM hand32、scene32分别比WAM-zero低>=10%，16步两域不差>5%；
  progress子集endpoint与32步两域不得比各zero基线差>5%。任一模型满足才PROMISING。

## 资源与停止

GPU0启动前检查空闲；拟合+WAM推理<=10min、额外产物<=100MiB，保留20GiB空盘。
无需重新提取0.99GiB token缓存。纯坐标标签/汇总CPU；所有模型运算GPU。
非有限、reset穿越、输入/label泄漏、非刚体旋转立即停。完成后记录实际显存、吞吐、ETA。

## Limitations / future evidence

预测旧策略未来不等于选择有回报的目标；数据含失败/静止段，progress报告避免掩盖。
单seed、少对象、同面板探索、预训练可见性与网络容量差异均限制结论。
若进入闭环，须测真实任务reward/holding和退步病例，不能把运动预测改善写成PPO提点。
WAM仍不接动作，不能从本实验推导动作反事实或终点的真实可达性。
