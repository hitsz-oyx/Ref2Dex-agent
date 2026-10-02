# HF21：结构化 Cm 的真实局部收益与重复决策

experiment_id: P-20261002-structured-contact-opportunity
family: HF21
probe_index_in_family: 2
kind: Decision / independent randomized execution Probe
status: ACTIVE

区分：联合后果驱动的可执行程序能否在真实H10控制中超过强Cup、直接评分及
打乱动作对照，还是预测改善尚不能转成控制收益。通过再设计持续控制/策略训练；
支持充分但失败停止当前配方，支持不足UNCLEAR且不追加seed或降低门。
最便宜方法是冻结已通过信息门的模型，短片段随机执行、重复观察，避免完整PPO。
依据[测量决定](../../decisions/D-20261002-structured-opportunity-measurement.md)。

模型checkpoint固定SHA3121c65f571210e71f761247a973adcabd2df029841777f7e81761e6edc533f0；
三个成员仅使用旧fit训练，归一化与全部网络冻结，不在新数据上训练/校准。
原native pilot、HF19/HF20标签/门和代码全部不改。新schema为
`ref2dex.structured_contact_opportunity_source.v1`，planning同名加`_planning.v1`。

固定科学seed611–618，assignment seed=seed+15000，proposer=assignment+30000。
96env、max300ticks，每first episode每early/clear最多2个H10窗口，冷却6tick。
current joint存在连续3步、rest高度>=5mm、warmup10、剩余episode>11、可完成H10；
不筛未来好坏、速度或任务结果。窗口内权重与起始旋转锚固定，六专家每步按新
观测反馈；片段外base。符合条件时重新观察、生成和随机分配。这种采集不是
纯Cm-on完整闭环策略评价。末步/重置污染窗口使该phase失败，不丢弃后继续。

8已知可执行候选仍生成，实际分配仅5臂：base0、Cup1、Cm2、direct3、shuffled4。
16slots `[0,0,1,1,1,1,2,2,2,2,3,3,3,4,4,4]`均匀独立抽样；合并概率为
.125/.25/.25/.1875/.1875。所有候选与冻结pre-only后果预测完成后才抽样。
未执行的随机候选5–7只有预测，不当真值，也不作为收益对照。
生成器32步/.15/梯度clip1，置信增益mean-1.645*std>0，几何loss差<=.02，
joint-support概率差>=-.05，当前/候选|z|>8返回Cup；全部保持既有实现。
direct只训练高度评分，选择风险来自对候选恒定的state-only；不使用未训练风险头。
state-only无法区分候选，保持Cup作为强实际对照。

保存每候选3成员冻结Cm的pre预测：支持高度减当前正高度(mm)、几何loss概率、
hand/object共同存在概率(event6+7)。Base使用真实自由旋转PD及law0，其余
使用真实锚定PD及law1，不能把base套入锚定生成缓存。全部完整NN在GPU计算。
实际Y为末3步支持（hand/object代理与whole-mesh CLR>=2mm）最小正高度减pre
正高度、H10曾CLR>=2mm后失去的几何loss、末3步hand/object共同存在。
force/mg>.1是存在代理而非识别手物接触对；H10不是最终稳定抓取或掉落验收。

主要估计器预先固定AIPW，mu是上述3成员均值，
`V_a=mean(mu_a+I(A=a)/p_a*(Y-mu_a))`。每臂使用同一Cm nuisance，但不同
候选动作/law；其偏差由真实分配残差校正，预测本身不是收益。原HT并列报告，
不依据哪个显著而选择主指标。最佳固定臂只在bucket<50的新fit部分按AIPW
高度选base/Cup，tie选base。group split固定SHA12651 motion/start，50–69 cal、
>=70 held；不在cal/held选择基线、模型或门。

零差噪声为base slots0 vs1、Cup slots2/3 vs4/5，分别以合并slot概率.0625/.125
校正同一臂Y-mu，两边augmentation抵消。高度门是max(1mm,2*abs(base零差),
2*abs(Cup零差))，两组都需实际分配；不是同状态精确重放。

fit/cal/held各>=150窗口/32episodes/8motion-start组、五实际臂各>=24；
cal与held每项比较两臂实际匹配各>=24窗口/12episodes/8组。比较Cm与Cup/state-only、
shuffled、always-base、fit选best-fixed、direct。所有比较在cal与held均需：
支持高度差>=本split噪声门；group和episode 1000次bootstrap90%下界>0
（seed12753/12754）；几何loss点差<=+.02；joint存在点差>=-.05；提议PD
不同Cup比例>=.10；实际Cm不同Cup窗口>=12。PD差阈值1e-5，实际比较在同一
所选轨迹上重算Cup PD，不冒充Cup反事实后果。完整报告改变步数、重复规划
episode与后续窗口数。所有门通过PROMISING，支持不足UNCLEAR，否则UNPROMISING。
bootstrap是机制Probe的探索判断，不能升级为最终策略因果Validation。

先排除seed623做新分配/前置预测完整原生工程，60tick、每stratum1窗口；
16slot枚举验证任意错误mu仍恢复期望、mu0=HT、完美mu残差0及重复臂期望零。
全部原生PD、mesh、rawforce、pre/post几何、RNG、32步规划、冻结全NN与预测
前置输入独立重放，不用工程结果改变科学seed或门。

预算<=3600秒/8GiB：准备固定预留60秒、新工程parent<=340秒，科学源+审计
<=3200秒；分析须在同一累计3600秒内。预留值是成本上界，实际工程另外报告，
既有slot1累计554.09秒和旧源/训练成本另报，不计为新数据收益或重复消费。
使用3张启动时确认空闲的GPU（当前4/5/6），CAMPAIGN总上限4；native内部600秒、
parent640秒、完整audit420秒/parent450秒，工程native240/parent250、audit180秒。
单phase native与audit顺序，科学panel三条GPU进程lane并行，无子代理。
任何phase/hash/时间/存储失败停止固定panel并保留，绝不静默延长或补seed。
CPU只做标签/统计/文件，旧checkpoint和外部项目只读，新cache/output仅本仓库。
