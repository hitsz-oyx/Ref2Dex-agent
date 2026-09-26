# Ref2Dex Research Debt

这里记录：

> 最终可能需要，但当前不会改变近期研究决策的实验。

Debt 不是 blocker。

不要自动偿还。

只有当它变成论文 claim、正式汇报或关键决策所必需的证据时，才升级为 active task。

---

## D001 — Self-trained baseline full reproducibility

Status: DEFERRED

已有：

`s1_airplane_lift`

307/320 = 95.94%

仍可补：

* 更多 random seeds；
* 不同 GPU；
* 完整 checkpoint reproducibility。

Why deferred:

当前 baseline 已足够支撑 Cm utility 探索。

Trigger:

最终论文正式报告 baseline 时。

---

## D002 — Multi-trajectory generalization

Status: DEFERRED

仍需：

* 更多 GRAB trajectory；
* object variation；
* trajectory transfer。

Why deferred:

最终 Cm policy utility 尚未成立。

如果核心方法本身还没成立，提前扩大任务分布的信息价值较低。

Trigger:

matched Cm-on/off 已出现稳定正向结果后。

---

## D003 — Cm statistical validation

Status: DEFERRED UNTIL POSITIVE PROBE

最终可能需要：

* 多 seed；
* matched initialization；
* fixed checkpoint；
* bootstrap / confidence interval；
* repeated evaluation。

Why deferred:

现在尚未找到明确正向的 Cm policy integration。

Trigger:

小规模 Probe 出现明确 policy gain 后。

---

## D004 — Long-horizon supervised rollout stability

Status: DEFERRED

Why deferred:

当前最终系统允许：

supervised/base policy + RL physical correction

因此 supervised decoder 不需要在进入 RL 前独立证明完美长时 rollout。

Trigger:

只有最终方法明确声称 supervised decoder 本身具备 long-horizon autonomous stability 时。

---

## D005 — Full original-Cmv2 failure decomposition

Status: DEFERRED

已知：

nominal hand flow mismatch 是重要误差来源，但不是全部。

仍可研究：

* embodiment shift；
* geometry mismatch；
* training distribution；
* coordinate representation；
* object motion target。

Why deferred:

完整解释错误来源不会自动产生 policy utility。

Trigger:

如果 adaptation Probe 表明 Cmv2 representation 本身可能值得继续，但某个具体 mismatch 成为 blocker。

---

## D006 — Complete paper ablation matrix

Status: DEFERRED

包括：

* reward-only；
* critic-only；
* actor representation；
* action ranking；
* teacher/distillation；
* alternative Cm dimension；
* different object representations。

Why deferred:

必须先找到一个有效主方法。

Trigger:

核心方法进入正式 Validation 后。

---

## D007 — Mixed MANO/Inspire geometry penetration audit

Status: DEFERRED UNTIL MIXED PRETRAINING IS A LEADING ROUTE

GRAB MANO 和几何重定向 Inspire 的手-物表面可能穿模；现有
`hand_min_object_distance_m` 为无符号最近距离，不能衡量穿透深度。
若重新训练使用这些数据，需要用物体网格 signed SDF 量化穿透，
对严重穿模帧做过滤/降权，并分别报告两种来源的保留率与效应。

Why deferred:

当前混合权重冻结迁移失败，64 样本仿真微调 Probe 也未通过继续门；
此时全面重建 81GB 几何缓存的穿模审计不会改变近期选择。

Trigger:

新 Cm 表示在仿真真实转移上出现正信号，需要重新使用混合预训练
来提高跨手迁移或样本效率时。

---

## D008 — Cm 辅助表征的置乱目标与多 seed 对照

Status: DEFERRED UNTIL A CLEARER POLICY-GAIN PROBE

2026-09-24 两训练 seed、两评估 seed 的训练期 Cm 辅助目标
Probe 合计 +5.86pp，但未过预注册 +8pp 升级门。若该路线
重新出现明确正向信号，正式 claim 需要同架构、同训练成本的
置乱 Cm 目标或非 Cm 辅助任务 placebo，以及更多训练 seed、
固定评估集和不确定性估计。当前做这些不会改变不升级该
固定配方的决策。

Trigger: 新目标/表示的预注册 matched Probe 达到升级门。

---

## D008 — Randomized action/contact mechanism validation

Status: DEFERRED UNTIL MECHANISM BECOMES A PAPER CLAIM

目前 seed151/152 的单轨迹、单自训练 actor、单腕部动作轴随机随访
发现一步上抬与五步接触保持存在反向权衡。正式机制 claim 仍需
其他训练 actor、动作轴、物体/轨迹和更多 seed 的固定协议复核；
最好直接比较 +0.1、0、−0.1 三臂以区别相对原策略的效应。

Why deferred:

这不会改变当前“单轴在线规则没有 policy utility”的近期决策。

Trigger:

如果最终论文要将该权衡作为核心因果机制而非探索性线索。

---

## D009 — 两步 Cm 结构的同数据对照

Status: DEFERRED UNTIL SEQUENCE CM BECOMES A POLICY ROUTE

当前 raw 拼接模型用 seed168 训练、seed169 测试；结构化
第一/第二步效应模型用 seed168/169 训练、seed170 测试。
它们分别通过/未通过各自判别门，但不能由这两个结果
单独证明“结构化架构优于 raw 架构”。若序列 Cm 成为最终
策略路线，需在同训练数据、同评估 seed、相同训练预算与
匹配容量下比较两架构的预测和策略效用。

Why deferred: 当前 wrist-x 效应不直接对齐抬升，重复 z
效应又未过物理门；该 ablation 现在不改变是否上线规划。

Trigger: 新任务对齐的序列 Cm 进入在线 matched Probe 后。

## D010 — Contact-supported credit representation after HF02

Status: DEFERRED UNTIL A NEW HF03-STYLE GOAL

HF02 的 canonical temporal expert-option Probe 已通过数据合同，但 held-lift
policy-value gate 低于 action-shuffled placebo（`-9.677pp`）。这排除了当前
“首次接触时直接选择 expert option”的具体 policy route，同时保留了接触阶段
短序列、executed handflow、contact-supported lift 之间存在机制信号的可能性。

未来若重新开启该方向，至少需要：

* 明确定义 executed-handflow、接触保持和承重标签的时间窗口；
* 先完成 CPU-only label/action-alignment audit；
* 使用新的 experiment ID、预算和 matched Cm-off substrate；
* 只有在新 seed 的物理方向稳定后，才设计新的 GPU Probe。

Why deferred:

当前 post-HF02 decision memo 选择冻结 Cm policy-utility campaign。不能通过
换 seed、horizon、metric 或模型头继续 HF02，也不能把本债务当作现有 Cm 增益证据。

Trigger: 用户建立独立 HF03-style goal，且新的 contact-supported credit 假设
成为论文 claim 或下一阶段核心决策所必需的证据。

2026-09-26 CPU-only preflight 已对 route-specific `motion_id=0` substrate
完成 label/action-alignment audit；post-handflow gate 为 `UNPROMISING`，未启动
PPO auxiliary Probe。该结果范围仅限单 motion，不能替代未来 HF03 的跨物体合同；
完整 provenance 见 [HF03 contact-supported credit audit](handoffs/HF03_CONTACT_SUPPORTED_CREDIT_AUDIT_20260926.md)。
