# Decision Memo：用随机候选后果学习 Cm 的局部价值适配器

当前需要决定的问题是：原生 uncertainty fallback 失败后，是否还有一个不改变 Cm
物理预测职责、且能区分“数据稀疏”和“价值转换失败”的最小路线。

关键证据：八个候选的 native PD 命令在每个窗口有明显差异（全候选两两距离中位数
约 `0.156`），因此动作覆盖不是候选坍缩；但 287 个随机窗口只覆盖 25 个
motion/start 组，每个组每个状态只观测一个候选后果。固定 ridge 的离线 score 适配
审计中，加入 Cm 物理特征后固定组留出 RMSE 仅从 `92.6` 降到 `90.0 mm`，五折
结果方向不稳定，说明数据稀疏和物理到任务价值的转换都仍可能是瓶颈。

选择的最小行动：在同一随机候选记录上训练一个固定 `Ridge(alpha=1)` 的
action-conditioned value adapter。输入为决策时刻 state、候选 PD target、候选 ID
和冻结 Cm 的 score/std/retention/release（含相对 fixed 的物理量）；目标为真实
one-candidate H10 physical score。模型只做候选后果的局部价值适配，不替代 Cm，
不预测最终成功率。按独立 motion/start 组留出，用已知 `p=1/8` 的 IPW policy
delta 与 fixed Cup 比较。

如果 Cm 输入的 adapter 在组留出中有正向且非跨零的 score 区间，下一步只做一次
新的 native A/B；否则关闭“同源 value adapter”配方，回到更高层 representation/
planning。停止条件是不扫描 alpha、阈值或 seed，不直接启动 PPO。
