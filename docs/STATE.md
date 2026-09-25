# Ref2Dex Current Research State

Updated: 2026-09-25

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

新增 12 条校正轨迹、10 个物体身份的 Cm-off 单策略续训 Probe：e300 在
两个评估种子为 16/128，续训前 e260 同输入为 13/128；至少 6 个身份
仍为零。四个已有自训练 checkpoint 的最优单模型仅 21/128，按物体
事后挑选的乐观上限 32/128。详见
[实验卡](experiments/probes/P-20260924-multitrajectory-baseline.md)。
duck 单物体 self-trained 续训在新种子206–208 为 **140/192**，源 e260
同轨迹/seed206 为 1/64；相同方法在 waterbottle seed209 为 0/64。
waterbottle 近接触重置比例退火续训 e340→e400 后，同 seed209 为
**7/64**，接触比例由 1.6% 升至 27.5%，未过预设的 16/64 与 30% 联合门。
12 轨迹参考动作控制只有 9/128，不能直接作为广覆盖 BC 教师。
自训练专家的仿真器物体身份固定路由在全新 seeds211–213 为
**55/192 (28.65%)**；airplane 17/45、duck 12/18、mug 9/15、
toothpaste 14/15 四个身份达到 25% 覆盖门，其余六类仍稀疏或为零。
这通过了探索性多轨迹覆盖门，但路由使用特权物体身份，仍不是
观测驱动的完整 GRAB 策略，亦不是 Cm 增益证据。
原始 `dataset/GRAB/data/grab` 当前有 1335 个 `.npz` 序列，
其中 268 个文件名包含 `_lift`；这次 12 条校正转换轨迹仅为一个
选定子集，不能报告成“整个 GRAB 数据集”的抓取率。上述计数只按
文件名清点，尚未完成可模拟性筛选。
初始策略观测的专家分类器在新 seed216 为 62/64，因 mug 仅 3/5
未过预设的四类零错误门。只调整 SVC 类别权重后，在新 seed217 为
64/64 并过探索门；该识别测试未将分类器接入在线控制，也未测试 Cm。
在线观测路由在新 seed218 的 64 个环境中与固定身份路由 **64/64**
专家选择一致，抓取为 **19/64**，固定路由为 **16/64**，过预设
保真门。相同 seed、轨迹和专家选择仍有 11 个逐环境抓取结果不同，
这两次仿真运行的差异不能解读为观测路由提升抓取。
五个自训练专家在 seed219 的同起点 Probe：固定路由两次为
20/64、23/64；五专家事后逐状态上限 33/64，固定 A 失败但
至少一专家成功 14 例，固定重复 B 补回 3 例。四类物体存在
不同专家的互补成功，过探索门；上限受重复运行波动影响，不能
当成可达的 Cm 成绩。
当前单右手 DExplore 过滤集合有 **59 条 lift-like 轨迹、29 类物体**。
张量和资产审计以及 64 环境仿真加载已通过；源 e260 在该池 smoke
为 3/64。59 轨迹 Cm-off 共享 actor 续训正在执行。原始 GRAB 的
1335 条序列和 268 条字面 `_lift.npz` 仍大于这个兼容子集。

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
- 新 s3 专家整段结果 Cm 的初始动作输入无增量：留出 AUC
  action-aware 0.7539、action-blind 0.7568、action-shuffled 0.7419；
  不升级在线路由。duck 上随机单步抬腕虽提高 H10 接触加权抬升
  10.10mm，但接触比例下降 1.38pp；预设联合门失败。
- 十步历史对 apple 的 H20 接触预测优于当前状态（RMSE
  0.332 对 0.382），但相对相同历史的无动作头 0.369 仅改善
  约 9.9%，未过预设 10% 动作信息门，且承重位移预测反而较差。

## 当前 blocker 与活跃假设

尚无同时满足“支持持续接触承重”和“Cm 能区分有效策略决策”的
动作族或监督目标。一步局部效应、简单序列、当前 V1.3 token 与若干
PPO 接法均未给出稳定 policy utility。duck 说明单物体策略可以学会
共享策略未覆盖的抓取。waterbottle 起点退火增加接触和少量抓取，
仍不足以支持继续同一局部课程。固定专家路由已证明四类物体
可以共同成功抓取；下一关键不确定性是能否从策略可用观测识别
专家，以及 Cm 是否在相同专家组合上增加决策价值。

## 下一步

完成 59 轨迹共享 actor 的 heldout baseline 评估；并在
五专家互补起点上收集可比较的 option outcome/action 数据，先做
action-aware、action-blind、action-shuffled 的离线 Cm Probe。
只有出现明显正向信号才投入在线 Cm 路由和正式 matched Validation。
未来论文需要
但当前不改变决策的实验见 [Research Debt](RESEARCH_DEBT.md)。
