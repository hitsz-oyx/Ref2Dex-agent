# Ref2Dex Current Research State

Updated: 2026-09-24

本文件是当前唯一默认事实入口；历史经过保留在
[阶段收尾记录](CM_CAMPAIGN_CLOSEOUT_20260924.md)、实验卡和
[旧版 STATE 归档](archive/STATE_20260924_precloseout.md)。

## 当前决定

用户已选择 [D-20260924-after-task-aligned-option](decisions/D-20260924-after-task-aligned-option.md)
的 Option A：本阶段停止新增 Cm 训练、仿真和 Probe，保留 checkpoint、
运行清单及报告，整理负结果。研究 Mission 没有改变；Cm 对策略的独立、
稳定增益仍未建立。这是当前 campaign 的资源决定，不是对所有 Cm 方法的否定。

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
PPO 接法均未给出稳定 policy utility。若未来重启，候选方向是把
接触维持到承重结果的时序 credit assignment 纳入表示/目标，并先用
小型 matched Cm-on/off/placebo Probe 判别；目前没有活跃实验。

## 下一步

按已选 Option A 完成现有报告和产物的复现边界核验，保存本地 Git
checkpoint，不启动新计算。若要改为高成本时序新架构，先依
[Decision Memo](decisions/D-20260924-after-task-aligned-option.md) 重新作路线决定。
未来论文需要但当前不改变决策的实验见 [Research Debt](RESEARCH_DEBT.md)。
