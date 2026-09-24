# Cm 研究阶段收尾（2026-09-24）

Branch: `agent/cm-cross-object`. Decision:
`docs/decisions/D-20260924-after-task-aligned-option.md`，用户选择 Option A。
本阶段停止新增计算；既有运行目录、checkpoint 与报告保留。

## 当前能陈述的结果

| 问题 | 证据 | 可陈述的边界 |
| --- | --- | --- |
| 自训练单轨迹抓取 | V1.28 固定整段路由在未见 seeds60–64 为 307/320（95.94%）；`src/task/CmResidual/docs/experiments/EXP-20260923-V128-ROBUST-SINGLE-TRAJECTORY.md` | 只覆盖 `s1_airplane_lift`；仿真器起始帧决定整段使用哪个 BC/PPO 专家，不是单一观测驱动网络。PPO 专家使用过 CmLite 奖励训练，且没有 matched 同路由 Cm-off 消融。 |
| 早期 CmLite 奖励正向信号 | 同训练 seed 的 CmLite e140 与 Cm-off e180 在 seed49 为 35/64 对 19/64；CmLite e140 在 seeds49–53 合计 185/320；`src/task/CmResidual/docs/activities/ACT-20260922-V122-CMLITE.md` | checkpoint epoch 不同，seed49 用于选择最佳 checkpoint，且 Cm-off 缺少相同五 seed 网格；只可视为探索信号，不证明跨 seed 的 matched Cm 增益。 |
| 一步预测进展作为 PPO 奖励 | V1.46 matched Cm-on 262/640，Cm-off 399/640，差 −21.4pp；`src/task/CmResidual/docs/experiments/EXP-20260923-V146-CM-TRAIN-ONLY-REWARD.md` | 否定固定 V1.37 模型、0.1 系数、真实接触门的联合配方。 |
| effect-rank PPO 配方 | `VAL-20260923-CM-EFFECT-PPO`：on 882/1536、off 789/1536、action-shuffled 1049/1536；`docs/experiments/validations/VAL-20260923-CM-EFFECT-PPO.md` | on−off +6.05pp，但训练 seed 间不稳、区间跨零；预注册联合正向主张 `REFUTED`，不能声称稳定 utility，也不能反向声称置乱显著更优。 |
| 原始 ObjectInteractionCm V1.3 的跨对象动作信息 | 冻结 checkpoint、五对象线性头留一：action-aware AUC 0.6387，action-blind 0.6694，shuffled 0.6382；`outputs/CmResidual/agent_cmv13_object_loo_s249/report.json` | `P-20260924-cmv13-object-loo-representation` 未过动作信息门；预训练索引的 train split 已包含五个物体，故不是整个 Cm 对未见物体的测试。10135 点比 1538 点仅 +0.0025 AUC，这是该目标的密度诊断。 |
| 任务对齐 grip+lift option | 单对象 self-trained e140、seed191、572 个有效状态：H20 grip+lift 相对 lift-only −8.20mm，环境聚类 95% CI [−14.08,−2.49]mm；`outputs/CmResidual/agent_s1_task_aligned_option_s191_h20_n128/effect_report_v2.json` | `P-20260924-s1-task-aligned-option` 为单 seed Probe，否定该固定动作族的继续门；不外推到所有接触维持策略。 |

## 科研结论

Objective A 在允许仿真器起始帧路由的系统设定下有单轨迹强阶段证据；
单一网络或不使用该起始帧的策略尚未由 V1.28 证明。Objective B 所需的稳定 matched
Cm-on/off 策略收益尚未建立；多对象泛化也尚未建立。正式 Validation
只对 effect-rank 的联合正向主张作 `REFUTED` 判定，其他探索性 Probe
只能保留各自的 `PROMISING`、`UNPROMISING` 或 `UNCLEAR` 结论。

最终报告或论文若坚持“Cm 改善抓取策略”的正向主张，现有证据不够。
本次停止当前 campaign 是研究资源决策，不等于证明 Cm 在所有可能的
表示、动作族或训练方法下都无效。未来若重启，需要另立时序目标与
matched 对照设计；当前不启动该项目。

V1.22 的窄范围正向信号与 V1.46 的负结果不是同配方复现：前者在
`s1_airplane_lift` 从零训练 seed45，使用 V1.22 CmLite、奖励系数 5，
两臂最佳 checkpoint 的 epoch 不同；后者在 s3 从同一 Cm-off e260
续训，使用 V1.37 CmLite、系数 0.1，并固定 e300 做新 seed 的
matched 评估。两者共同说明当前尚无稳定、可迁移的 Cm 奖励收益，
不能由前者推出普遍有益，也不能由后者推出所有 Cm 奖励必然有害。

## 复现入口

V1.28 的逐 seed 运行 manifest 在
`outputs/CmResidual/agent_v128_robust_router2_s{60..64}_n64/`。
effect-rank Validation 的父 manifest 和完整矩阵在
`outputs/CmResidual/cm_effect_validation/`。
V1.3 表征 Probe 的 `run_manifest.json` 固定模型、校准和 transition SHA；
单对象 option Probe 的 `run_manifest.json` 固定 actor、motion、随机种子、
动作剂量及 transition SHA。以上产物均未合并为新的正式科学 claim。

## 本地产物核验（2026-09-24）

逐一读取 V1.28 的五份 `run_manifest.json` 与 `rollout/summary.json`：
seed60–64 均为 `COMPLETED`，各有 64 个 episode，成功数依次为
61、62、63、59、62。正式 effect-rank Validation 的父清单列出
12/12 训练与 72/72 评估；12 个训练 checkpoint 和 72 份子运行清单
均存在，子运行均为 `COMPLETED`。按子运行重新求和为 effect-rank
882、action-shuffled 1049、off 789，与 `analysis.json` 一致。
V1.3 表征 Probe 的 `report.json` SHA256 与清单一致；单对象 option 的
`effect_report_v2.json` 为 `UNPROMISING`，其效应点值与区间和上表一致。
从原始 transition 离线重算单对象 option，287/285 两臂计数、−8.198mm
主效应及环境聚类区间与报告完全一致；无接触权重的 H20 位移差也为
−14.49mm，两臂重置率均为零。
五个 V1.28 运行均使用同一份 38 项起始帧路由表。评估代码在 reset 后读取
`task.start_times`，按最近起始帧选定 BC/PPO 专家，整段 episode 固定路由；
五份 summary 的在线 proposal 选择率均为 0。故 307/320 是这个组合系统
在该仿真设置下的成绩，不能表述为单一 actor 或在线 Cm 选择的成绩。
按固定路由表重算各 episode 的已选专家结果：BC 路由为 166/171，
PPO 专家路由合计 141/149。两类 episode 的起始帧分布不同，
这些分组成功率不能当作 BC 与 PPO 或 Cm-on/off 的因果比较。
V1.28 活动记录记载路由先在 seeds49–53 构建，再用已揭盲的
seeds54–58 调整；这些探索运行的清单时间早于 seeds60–64，
后五份清单的路由表完全相同。现有运行清单支持规则在五个评估 seed
期间保持固定；尚未找到 seed60 运行前独立封存的最终路由表文件。

这是已有报告及产物完整性的核验，没有重跑仿真、训练或统计分析；
不能据此扩大任何科学结论的适用范围。
另核对 V1.46 的 20 份评估子清单均为 `COMPLETED`，按子清单求和
Cm-on 262、Cm-off 399，与父分析一致；seed 聚类 95% 区间为
−27.34 至 −14.84pp。
对当前磁盘文件重新计算 SHA256：V1.28 路由的 BC、CmLite 与五个
PPO 专家，V1.3 冻结 Cm，以及 V1.46 的共同源、Cm-on 和 Cm-off
末端策略共 11 个 checkpoint，均与对应运行清单一致。此检查只确认
文件身份，没有重新执行训练或评估。
