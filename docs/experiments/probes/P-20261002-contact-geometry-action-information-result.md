# HF19 slot1：H10 有信息，随机组合选择未产生增益

结论：`UNPROMISING`，slot1后关闭本配方，不启动slot2、PPO或完整成功率比较。
原[协议](P-20261002-contact-geometry-action-information.md)与
[模型合同](P-20261002-contact-geometry-model-contract.md)冻结不改。
此结果不反驳Cm核心思想，也不代表完整目标已完成。

GPU1完成12个真实采集seed571–582：3745个H10窗口、37450个真实转移。
fit/cal/held为1899/989/857窗口；held243episodes、22个motion/start组，
651early/206clear。base/cup/随机组合支持门通过，工程seed570未用于拟合。
[全部源审计](P-20261002-contact-geometry-source-audit-r1.json)复算PD、反馈、
概率、rawforce/mesh联合标签与全部pre/post几何；mesh/任务标签误差0。

四条件cm/state_only/shuffled/direct_score，各三个seed12601–12603、1000更新。
所有模型在GPU上从头拟合，实际耗时130.58秒，无追加epoch/seed/阈值。
held一步物体dv RMSE cm .14538、state .14649、shuf .14640m/s；相对位置
RMSE11.104、11.100、11.124mm，均未达到原双控制5%改善门。
H10联合支持高度MAE15.656、18.250、18.272mm，约14.2%改善；形成支持抬升
Brier .06987 vs .08075/.08159。部分任务后果信息成立，不能替代失败的整体门。

[独立拟合审计](P-20261002-contact-geometry-fit-audit-r1.json)验证全部52标签、
逐步历史/当前原始力/真实PD输入、fit-only归一化、12个模型的更新合同。
保存观测上的NN指标与全部候选预测回放误差0，actual候选输入误差0；
归一化最大误差4.77e-7。世界坐标与native几何的浮点差异按数值精度检查。
此前一个把统一绝对误差限用于大幅归一化delta的纯工程检查失败，原代码和
[失败记录](P-20261002-contact-geometry-model-input-audit-failure-r1.json)保留；
修复相对float32容差，未修改科学门、标签或网络，成本已计入。

fit选出的最佳固定参考是rotation_cup。留出Cm选择器改变参考158/857窗口
（18.44%），actual策略匹配154窗口/104episodes/21组。估计收益比较：

| 对照 | Cm支持高度差mm | motion/start组90区间 |
| --- | ---: | --- |
| always-base | +17.277 | [10.133,24.666] |
| 最佳固定cup / state-only | −0.089 | [−0.645,+0.404] |
| action-shuffled | +1.532 | [−0.839,+3.443] |
| 直接任务评分 | +0.078 | [−0.648,+0.916] |

这说明本配方尚未使有条件的动作选择优于强固定程序或直接评分。
随机组合池held平均−4.003mm，cup+10.994mm；前者是随机池策略，不能称
某个逐状态候选的真实regret。选择器大部分保留cup，不能把胜过base归功于
有增益的Cm修正。held重复cup槽差−11.696mm，原噪声阈值23.393mm；原门
保留，且即便忽略阈值，vs强参考的差/区间仍不支持正向选择增益。
原联合存在point门相对base亦失败，只报告IPW估计，不解释成正式掉落因果率。

[选择结果](P-20261002-contact-geometry-opportunity-result-r1.json)及
[独立选择审计](P-20261002-contact-geometry-opportunity-audit-r1.json)保存
float64候选谓词、实际物理score、propensity及组/episode区间复算，误差0。
这里只评价随机记录中的离线策略机会；没有真实执行完整Cm重新规划策略，
没有未执行候选真值，没有正式稳定抓取或策略训练收益。

下一步改变候选生成机制：让冻结Cm的H10物理后果预测主动生成受约束程序，
与强cup及同预算直接评分/shuffled生成器做真实随机机会检验，而非继续训练
当前随机组合排序器或调整它的门。见[下一路线决定](../../decisions/D-20261002-cm-optimized-action-generation.md)。
