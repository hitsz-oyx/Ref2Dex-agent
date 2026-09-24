# Ref2Dex Current Research State

Updated: 2026-09-24

本文件是当前唯一默认事实入口；历史经过保留在
[阶段收尾记录](CM_CAMPAIGN_CLOSEOUT_20260924.md)、实验卡和
[旧版 STATE 归档](archive/STATE_20260924_precloseout.md)。

## 当前决定

用户已于 2026-09-24 重新授权多轨迹 baseline 训练及 Cm Probe；
[执行选择](decisions/D-20260924-reopen-multitrajectory-cm.md)取代此前的
暂停决定。北京时间 2026-09-25 10:00 前，常规研究路线选择由 AI 自行
决定并记录。Mission 与资源硬上限不变。

## North-star 状态

| 目标 | 当前证据与边界 |
| --- | --- |
| Self-trained grasp | 不使用官方 actor checkpoint 的单轨迹固定路由在新 seeds60–64 为 307/320（95.94%）。路由读取仿真器起始帧，选择整段使用的 BC/PPO 专家；不是单一观测驱动 actor 的成绩。 |
| Cm one-step information | 若干随机动作干预中存在可学物理效应，但冻结 V1.3 token 的五对象线性头留一 Probe 未显示动作信息增量；结论依赖表示与分布。 |
| Cm policy utility | **尚未证明**。effect-rank 正式 Validation 的联合正向主张 `REFUTED`；其他已测试接法的 Probe 未建立跨训练 seed 的稳定 matched 增益。 |
| Generalization | 自训练策略对未见物体的持握抬升弱，未见物体上的 Cm 策略收益未建立。 |

## 决定下一步的事实

- V1.28 的 307/320 是固定路由系统结果。路由中的 PPO 专家使用过
  CmLite 奖励训练，但没有同路由 matched Cm-off 消融；在线 Cm 选择率为零。
  早期 CmLite 奖励的 35/64 对 19/64 仅为 checkpoint epoch 不同、
  缺少相同多 seed Cm-off 网格的探索性正向信号。
- V1.46 一步预测进展 PPO 奖励的 matched 结果为 Cm-on 262/640、
  Cm-off 399/640。`VAL-20260923-CM-EFFECT-PPO` 的 effect-rank 为
  882/1536、off 789/1536、action-shuffled 1049/1536；effect−off
  +6.05pp 且区间跨零，真实动作对应的联合正向主张未过预设门。
- 五对象冻结 V1.3 token Probe：action-aware 留一 AUC 0.6387、
  action-blind 0.6694、action-shuffled 0.6382，未过动作信息门。
  留一仅作用于新拟合的头；冻结 checkpoint 的预训练数据已含五个物体。
- 任务对齐 grip+lift option 在单对象 self-trained e140 的 H20
  接触加权抬升相对 lift-only 为 −8.20mm，环境聚类 95% CI
  [−14.08, −2.49]mm，停止该固定动作族。

## 当前 blocker 与活跃假设

尚无同时满足“支持持续接触承重”和“Cm 能区分有效策略决策”的
动作族或监督目标。一步局部效应、简单序列、当前 V1.3 token 与若干
PPO 接法均未给出稳定 policy utility。活跃 Probe 是先建立
[12 轨迹 baseline](experiments/probes/P-20260924-multitrajectory-baseline.md)，
再针对接触至承重的时序信息设计 matched Cm-on/off/placebo Probe。

## 下一步

运行 12 轨迹 self-trained Cm-off 续训并逐轨迹评估；依据覆盖情况决定
扩展数据、修正训练分配，或进入 matched Cm 机制 Probe。未来论文需要
但当前不改变决策的实验见 [Research Debt](RESEARCH_DEBT.md)。
