# HD03：生成程序的接触预测与决策一致性

experiment_id: P-20261002-generated-contact-consequence-diagnostic
family: HD03
probe_index_in_family: 1
kind: Decision / existing-source diagnostic
status: ACTIVE

问题：冻结Cm的生成程序未证明超过cup，主要待区分的是后果模型已经提示
接触保持会变差但决策没有使用，还是在优化后动作分布上的预测不可靠。
结果决定下一步优先补全物理后果的决策约束，或重做模型的动作效果/支持分布。
最便宜方法是用现有已全审计源及冻结模型重放cal，不新增采集或训练。

HF20原Probe保持UNCLEAR：held334，其中Cm23匹配，低于24；vs cup
HT支持高度−6.372mm，组90[−14.214,+.916]；没有成功收益或负因果定论。
实际16Cm窗口/160步不同cup，原收益/风险/噪声/支持门不改，不补seed。
本诊断不能把HF20升级PROMISING，也不直接进入PPO/完整成功率。

只分析SHA12651 bucket50–69的cal306窗口；held不用来调模型或决策参数。
所有候选只用当前history/native/force/mg/geometry/真实初始PD及已知反馈律，
重放冻结cm/state-only/shuffled/direct-score三成员网络；未来只作实际分配标签。
非执行候选只有模型预测，没有反事实真值。source/模型/归一化/依赖全部hash固定。

报告：生成Cm相对cup预测H10手/物末3步存在概率、联合事件的Fréchet上下界、
几何损失、支持抬升概率及支持高度；若生成候选的joint概率上界低于cup下界，
模型自身已经提示接触风险。检查lift事件概率是否超过手/物存在概率（集合包含
关系）；这是预测内部一致性，不把预测当真值。实际分配标签分别报告手/物存在
Brier、H10几何损失Brier和支持高度误差/偏差，并报告各arm观测数。
候选及其forecast用完整冻结NN验证，与已保存的生成器预测复现一致。

同时报告HT权重质量及已观测conditional rates，解释有限分配噪声；这些rate
不是随机总体的因果效应，也不替代原HT门。后续若使用不同效应估计器，必须另行
固定新卡，不能事后给此处或HF20换标签。

单空闲GPU1，<=600秒/32MiB，无训练/梯度优化/新rollout。当前会话直接执行，
不创建子代理、不改Mission/claim或资源边界。正式收益仍需新独立Probe和后续
策略训练/稳定抓取验收，完整goal保持ACTIVE。
