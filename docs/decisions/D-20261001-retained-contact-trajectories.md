# 从平均抬升信号推进到保留后果与相对动作效果

2026-10-01，Decision Checkpoint。HF09 的3/3预算保持关闭，新问题为：能否把已
观察到的短期抬升信号变成保留接触/进度的动作选择，而不是牺牲掉落风险？
不改变 MISSION、任务、claim、资源或外部权限。HF09有信息进展，此处不是重置
失败Probe预算继续调旧阈值。

证据：真实随机介入支持平均抬升+5.106mm，但末3步保留抬升+2.901mm区间跨零；
35.1%的Cm正均值窗口未保留进度，已有抬升窗口掉落11/140vs4/122。
不能只用瞬态正向高度均值或全部窗口稀释过的掉落门，决定进入PPO。

选择HF10：Cm预测10步逐步物体高度变化/接触概率、末3步联合接触和抬升后
release风险。明确预测base轨迹，并用共享非线性分支的candidate−base差学习
动作影响，base效果严格为零。得分由末3步最低高度增益和联合接触计算；
相对base的风险上界不得增加，接触保持下降最多.02。不用长期V。
release标签覆盖窗口内新抬升随后掉落；评价不按处理后的“曾抬升”筛选，始终
报告全部状态及处理前已抬升分层，避免选择偏差。原HF09标签保持原定义。

保留旧Cm encoder/归一化作为初始化，使用已记录实际仿真转移适配新物理heads，
无future action/label输入。fit/calibration仍按冻结initial-frame组，旧held只作
已使用证据的工程审计；不声称重新独立验证。state-only六slot与action-shuffled
同训练预算、同物理目标，best-fixed仅用uniform-fit真值选择。

HF10最多2个Probe：slot1旧数据物理轨迹学习/校准（9models×1000updates，单GPU，
预计2分钟）；slot2固定新seed351–356的5-selector推荐池随机决策（Cm/state-only/
shuffled/base/best-fixed），已知动作概率合并重复推荐，候选2+base8后重新观察。
该前瞻数据同时检查四控制、真实动作改变与相对物理效应；单决策混合历史效应
仍不能替代完整pure-policy或训练所得actor收益。每slot≤60分钟/8GiB，未知GPU
进程不动。所有工程尝试计入相应slot，输入/动作合同失败立即停。

正向后设计独立策略训练及稳定抓取验证；风险、coverage或四控制不通过则停止
这条retention机制局部调参，复盘表示/候选控制片段。模型有限/能执行不等于
有用；旧mean-lift正向信号也不自动满足最终claim。无新外部授权需求。
