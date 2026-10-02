# HF23：接触约束真实介入，未建立风险收益

source_experiment_id: P-20261002-contact-risk-interventions
run_id: P-20261002-contact-risk-interventions-source-r1
run_status: COMPLETED
Probe classification: UNPROMISING

固定8seed651–658全部完成，5440执行前检测中88接纳H10窗口（1.618%），76个
episode、40个初始motion/start组。actual base/Cup/Cm/direct/shuffled分别
4/6/34/33/11，primary支持门通过。12个episode重新观察并再执行，共12个后续
窗口。Cm实际34窗口/340步PD不同于同实际观测上的unguarded；这是指令改变，
不是未执行direct轨迹真值或逐状态regret。

| Cm相对direct/state-only | AIPW点差 | group90区间 |
|---|---:|---:|
| H10任一步接触丢失 | +1.385pp | [-12.269,+16.563]pp |
| 支持抬升变化 | -4.417mm | [-17.711,+8.471]mm |
| 几何掉落 | -13.161pp | [-32.339,+6.064]pp |
| 末3步joint存在 | +1.126pp | [-12.682,+13.628]pp |

接触丢失未减少5pp，置信区间跨0；高度点差低于-2mm且区间下界低于-5mm。
重复同一direct程序的随机slots接触零差-21.217pp，Cm零差-3.722pp，固定噪声
门42.434pp亦未过。不是同状态精确重放，不证明Cm产生负因果效应。
原HT并列：接触差-3.030pp、支持高度-17.343mm，没有事后选择估计器追门。
几何掉落和joint点非劣门通过，但不替代失败的主要风险/高度门。

所有控制完整保留：相对base高度-0.170mm/接触+14.051pp，相对shuffled
高度-26.027mm/接触+11.447pp，区间跨0。相对Cup高度+56.626mm/接触-57.483pp，
但只有6个真实Cup窗口，不能替代强direct/state-only对照。AIPW估计可超出概率
范围，须按估计差及区间解释。fit无实际base，按原fit-only规则best-fixed不能
选定（null），明确保持未完成；不偷偷从全数据选择强参考。
15个初始clear窗口，仅Cm5/direct7/shuffled3，base/Cup为0；不据此声称稳定
抓取或可靠已抬升后掉落收益。完整策略/训练所得actor/最终45tick保持仍未证明。

全部8native与8完整audit exit0；全检测优化/RNG/当前输入/完整NN/nuisance/
专家冻结以及实际每步PD/rawforce/mesh/pre-post几何独立通过。独立统计原始
支持高度最大差7.15e-6mm、统计最大6.44e-7，原门分类完全一致；额外独立
标量PD审计确认34窗口/340步改变，没有依赖分析实现的动作计数。

累计预算2731.752秒（45.53min），包含失败343.856秒、两次有效工程、计算
修正保守10秒、准备60秒、旧静态工程16.456秒与科学源/所有分析。科学source
GPU44.863min、audit GPU33.821min，3设备lane wall与GPU累积分别报告；旧
模型/源成本另报。forced工程p=1窗口全部排除科学及训练，不使用其后果选门。

决定：关闭HF23 1/1当前风险约束配方，不补seed、调阈值、追加适配或启动本
配方PPO。HF22预测信息子门保留但不升级；C3仍OPEN。下一步返回更高层候选
控制职责，检验随真实手物相对运动反馈的执行程序，而非继续在固定专家凸混合
上约束一个未建立可靠动作风险差的模型。

证据：[固定卡](P-20261002-contact-risk-interventions.md)、
[完整结果](P-20261002-contact-risk-interventions-results-r1.json)、
[统计审计](P-20261002-contact-risk-interventions-statistics-audit-r1.json)、
[PD改变审计](P-20261002-contact-risk-interventions-actuation-audit-r1.json)、
[成本与源审计汇总](P-20261002-contact-risk-interventions-completion-r1.json)。
