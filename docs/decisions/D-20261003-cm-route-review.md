# Decision Memo：关闭回顾性价值校准，改做决策时刻 Probe

当前需要决定的是：P-20261003 的离线 relative-value head 在 held 上很强，是否足以进入
native 控制和策略训练。

关键证据：原 head 的 12 个特征包含 H10 内第 2--10 个真实干预后状态的 Cm 预测；在线
控制器只能看到当前和历史，因此原 `PROMISING_CALIBRATION` 不是有效决策门。严格审计中，
只用当前决策时刻特征的 held Spearman 为 0.757（raw 当前 Cm 为 0.717），而回顾性 head
为 0.927；463 个 native 窗口的实际校准控制相对 fixed-Cup 仅 +0.77 mm，组级 90% 区间
[-12.43,+18.39]，原 Cm 相对 fixed 为 -1.55 mm。

选择的行动是关闭回顾性 head/native 配方，不启动 PPO，也不扩大普通 Cm 数据；用现有
HF16 记录做一个严格的决策时刻 current-step relative-value Probe。新 head 只使用触发
时的当前 Cm 候选预测、初始状态、初始 clearance 和接触位，目标为真实 H10 保留高度减去
同一触发时刻的 fixed-Cup 预测。fit/held 仍按固定 motion/start group 分隔。held 上相对
raw 当前 Cm 同时提升 Spearman 至少 0.10、RMSE 至少 10%，才做有界 native follow-up；
否则整个 local-value conversion 路线暂停，回到更高层 representation/planning 复盘。

成本是 CPU 小型 ridge 审计；native follow-up 最多沿用单 GPU 少量 seed。失败关闭这一具体
current-step 配方，不扫描阈值、ridge 或 seed，不进入策略训练。

该 follow-up 已完成：5 个 seed、384 个窗口支持充分，但 calibrated-vs-fixed 的
90% 组级区间仍跨零，且 calibrated-vs-Cm 的 joint 控制门失败。因此 current-step
配方也关闭，C3 不升级，下一步必须换 representation/planning 或动作优势识别合同。
