# Decision Memo：验证 Cm 的不确定性回退候选排序

P-20261003 的预注册 Cm selector 没有通过：它在 287 个随机化窗口中改变 282 个
决策（98.3%），相对 fixed Cup 的 IPW score 90% 组区间为 `[-7.1,+45.1] mm`。
但同一冻结 Cm 预测上，一个不再拟合数据的安全使用规则给出局部信号：只有当候选
score 超过 fixed Cup 两倍 ensemble standard deviation、candidate 没有 OOD、retention
不低于 fixed-.05、release 不高于 fixed+.05 时才执行，否则回退 fixed Cup。

这个固定规则在原随机 assignment 的 known-propensity 离线重放中改变 143/287
决策（49.8%），score IPW 增益 `+24.0 mm`，90% 组区间 `[+7.0,+40.7]`；retained
height、末三步 contact 和 clearance 的下界分别为 `+0.107`、`+0.033` 和 `+0.107`。
这是 PROMISING_LOCAL_SIGNAL，而不是 policy utility 或 stable grasp 结论。该结果
来自同一批数据的冻结模型重放，不能直接启动 PPO。

选择的下一步是 fresh native validation：固定该 2-sigma 规则，在新的 contact states
上与 fixed Cup 以已知概率随机分配，仍执行 one-tick candidate + nine-tick fixed Cup
continuation，独立记录真实 score/contact/clearance。只有 fresh native 的 score 和
风险门同时通过，才进入小规模 categorical option PPO；Cm-on/off 将共享 actor 初始
权重、环境预算和 reward，on 只额外接收 Cm 的物理 candidate features，off 使用同形
零输入和 fixed-Cup fallback。

停止条件：fresh native 支持不足、score 区间跨零或风险下界失败时，关闭此 fallback
配方，不扫描 sigma、阈值或 seed；保留当前 positive local signal 作为未完成机制证据。
