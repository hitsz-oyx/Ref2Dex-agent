# P-20261003-relative-task-value-calibration

Family: HF26；类型：Decision Probe；状态：CLOSED；离线初始标签曾为
`PROMISING_CALIBRATION`，时间信息审计后撤回，native follow-up 为 `UNPROMISING`。

这个 Probe 只检验物理预测到任务价值的转换，不改变 Cm 的真实动作后果预测职责。
输入是 HF16 `native_pd_direct_control` 的 12 个 seed、903 个随机 H10 窗口。Cm 的
三成员 motor-conditioned one-step 预测、候选动作和强 fixed-Cup 参考全部冻结；不加入
普通 Cm 数据、不运行仿真、不训练 actor/PPO。

每个窗口先从 frozen Cm 预测中得到实际候选相对 fixed-Cup 的高度、接触保持、释放风险
和 ensemble spread，再训练一个 ridge head 预测连续的
`delta-U = actual_H10_retained_height - frozen_model_fixed_prediction`。目标保留真实
动作后果，使用实际 H10 保留高度而非成功标签。fit/cal/held 按固定
`sha256("9851/motion/start")` 个位桶分成 `0--3 / 4--5 / 6--9`；ridge λ 固定为 1，
不用 held 选超参。初版实现使用百分位桶，导致本数据的 fit/cal 为空，已标为无效实现并
修正后重跑。

预注册判定：held 至少 100 行且至少 8 个 motion/start group；校准 head 相对 raw Cm
relative score 的 held Spearman 提高至少 0.10，且 held RMSE 降低至少 10%，两项都通过
才标记 `PROMISING_CALIBRATION`。支持不足为 `UNCLEAR`；任一转换门失败为
`UNPROMISING`，不启动 native follow-up 或 PPO。该 Probe 不声称反事实 regret、完整
成功率或稳定抓取。

补充审计发现：原 12 维特征的均值/极值跨越 H10 内第 2--10 个真实干预后状态，在线
决策不可获得；决策时刻重建后 held Spearman 仅 0.757，不能支持原校准门。保留的 7-seed
native follow-up 共 463 窗口，校准相对 fixed-Cup +0.77 mm，90% 组级区间
[-12.43,+18.39]，因此关闭该具体配方，不进入 PPO。
