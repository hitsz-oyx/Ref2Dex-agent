# Decision Memo：关闭 Cm uncertainty fallback 原生验证路线

当前决定的问题是：离线 known-propensity 重放中看似正向的 2-sigma uncertainty
fallback，是否值得进入策略训练。

关键证据是五个新 native seed 的固定 A/B：190 个窗口，Cm 与 fixed Cup 各 95 个，
支持门通过；但 score 差为 `-2.79 mm`，90% motion/start bootstrap 为
`[-15.74,+11.00] mm`，retention/contact/clearance 的区间下界分别为 `-0.290`、
`-0.244`、`-0.295`。离线正信号未迁移，且安全相关局部指标没有改善。

因此关闭当前 uncertainty fallback 及其 categorical option PPO 入口。保留冻结
Cm 的一步物理信息和离线候选审计作为机制证据；不扫描 sigma、阈值或 seed，不扩大
普通 Cm 数据，不把本 Probe 写成 Cm 核心假设或最终成功率的反证。

成本为五次短 native rollout 和一次审计，全部记录、checkpoint hash、PD 执行合同
保留。若未来继续 C3，应改写更高层的 representation/planning 或动作优势合同，
而不是在当前 selector 上继续追门；本决定不需要新的外部授权。
