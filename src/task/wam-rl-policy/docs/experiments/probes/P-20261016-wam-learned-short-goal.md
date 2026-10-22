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
status: UNPROMISING
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

## 实际结果与审查

运行代码 `c85a1837`，两臂均完成固定1000步，未按留出成绩选模型。
产物 `outputs/wam-rl-policy/learned-short-goal-20261016-r1/`：
`manifest.json`、`summary.json`、两臂checkpoint、`predicted-goal-forecasts.pt`、`audit.json`。
模型计算总51.69s，GPU0采样显存最高1167MiB，产物13MiB；运行完成并释放GPU。

以下均为8个留出物体的对象宏平均、单位mm；448状态，其中146个progress状态。

| 终点条件 | endpoint48全体 | endpoint48 progress | WAM hand32 | WAM scene32 |
|---|---:|---:|---:|---:|
| 零位移 | 66.73 | 136.21 | 47.65 | 47.50 |
| 按剩余时间插值 | 101.76 | 80.93 | 56.84 | 57.15 |
| raw-query预测 | 73.32 | 118.82 | 45.57 | 48.26 |
| wam-mean预测 | 67.61 | 103.05 | 42.54 | 44.58 |

两臂均未过预设gate，当前固定方案 **UNPROMISING**。wam-mean的fit endpoint18.87mm、
held67.61mm存在明显泛化差距；held translation误差64.00mm，不能把问题主要归于旋转。
这不能区分数据不足、混合行为多义性、输入表征和拟合方式，也不构成WAM路线的负结论。

保留的局部信号：wam-mean目标下WAM hand32/scene32比零目标改善10.7%/6.2%；
progress子集两域改善29.2%/27.3%。相同预测终点的几何插值hand32/scene32为60.48/47.85mm，
WAM为42.54/44.58mm，提示条件手物协同信息在非oracle目标下仍有作用；不代表PPO收益。

审查：10项相关单测通过。独立audit重载两模型，在相同TF32精度与128 batch下
预测逐值重放一致；对象fit/held不交叉、合法旋转、刚体点变换与全留出误差重算通过。
最初审查用不同精度/64 batch出现最高0.102mm位移差，统一运行精度与batch后归零；
未改模型或放宽gate。审查仅验证保存/计算关键链路，不能认证目标可达或预训练因果贡献。

## 结果后的 Decision Note

- 决定：不追加旧混合行为终点回归预算，也不把本次失败改判为通过；不启动原定的
  “合格预测终点直接接原生PPO”分支。保留当前模型用于诊断。
- 理由：旧行为终点预测不是任务目标选择；即使预测准确，也没有回答机器人如何实现它。
  当前代码 `Actor(reference-gaussian)` 仍加BC动作均值，`shadow_loop`每拍调用BC，
  因而此前full-action实验也不能代表从零、无BC依赖的PPO路线。
- 下一项最便宜的blocker工作：建立独立动作入口和明确的运动参考观测合同，先做原生
  工程smoke；之后另立Decision Probe，检验同一因果短目标下WAM参考是否让PPO更易执行。
  短目标应受速度/位移约束并在明确控制间隔内固定，由真实任务回报选择，不能把旧轨迹
  回归误差作为唯一资格。新实验须声明目标来源，不能用真实未来作在线目标。
- 比较对象：同目标WAM参考、同目标几何参考、task-only；共享原生动作空间和交互预算。
  先检验真实状态转移与任务保持，再决定是否扩训练。辅助tracking不能替代任务结果。
  WAM无动作输入，禁止把其预测误用为候选动作反事实打分器。
- 此轮到此停止新增训练。下一轮先完成接口smoke及单独有界协议，出现非有限、目标/动作
  坐标错配或干预未进入执行即停。未改变MISSION/claim，不需要新外部授权。

相关方法边界见[运动先验到RL的一手文献审计](../../../../../../paper/point-motion-goal-policy/2026-10-16-motion-prior-rl-bridge.md)。
Dream2Flow的flow-reward SAC支持无BC桥接的可行性，但不是通用小数据PPO证据；
Track2Act的共享残差控制器依赖机器人示范BC，不能直接当作我们的既有执行桥。

## 2026-10-22 负结果 CPU 审计：结论范围修订

这是c85a1837的旧行为endpoint回归+冻结旧WAM预测诊断，没有新自主闭环/成功率评估，非v6执行失败。448held状态来自8物体且窗口相关；不等于448独立任务。保留endpoint门槛未过及部分hand/scene改善，不能由此否决未来预训练对RL的样本效率收益。预测旧行为未来不是任务最优目标或可达性认证。

本追加不改变原始数字、gate或运行产物；完整身份、版本边界与排除项见[逐卡证据审计](../../../../../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-card-audit-20261022.md)。
