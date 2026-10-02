# P-20261003-current-decision-task-value

Family: HF27；类型：Decision Probe；状态：COMPLETED，离线 `PROMISING`，尚未做 native follow-up。

HF26 的回顾性校准被时间信息审计撤回：它使用了触发后的 H10 诊断摘要。HF27 只取每个
HF16 窗口第一个决策时刻的 Cm 候选预测、初始高度、初始 clearance 和接触位，训练一个
冻结 ridge head 预测
`actual_H10_retained_height - current_decision_fixed_Cup_prediction`。这保留 Cm 的
动作后果预测职责，不是成功率分类器；不加入普通 Cm 数据，不运行仿真或 PPO。

固定 motion/start group 切分得到 fit/cal/held = 401/102/400，held 9 组。相对 raw
当前 Cm score，校准 head 的 held Spearman 为 0.691（raw 0.429），RMSE 为 41.03 mm
（raw 56.37 mm）；两项预注册门（Spearman +0.10、RMSE ratio <=0.90）均通过，标记
`PROMISING`。这只是决策时刻的事实结果排序 Probe，不能证明反事实动作优势或策略效用。

下一步若资源允许，只用这个 current-step bundle 做一次少量 native matched panel；若
calibrated-vs-fixed 的组级局部效用未同时通过 +2 mm、90% 区间为正、loss/joint 控制门，
关闭 local-value conversion 路线，不启动策略训练。
