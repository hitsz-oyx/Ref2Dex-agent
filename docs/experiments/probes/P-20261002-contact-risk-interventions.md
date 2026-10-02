# HF23：接触风险约束的实际介入

experiment_id: P-20261002-contact-risk-interventions
family: HF23
probe_index_in_family: 1
kind: Decision / independent randomized local-control Probe
status: ACTIVE

区分：冻结Cm接触风险能否在实际约束介入状态减少H10接触损失并保留抬升，
还是已学到的风险信息无法改善动作。若通过，再固定更完整对照及持续策略/学习
方案；支持充分失败停止此约束配方，支持不足UNCLEAR，不补seed、调门或重训。
最便宜方法是预先检测介入后随机实际执行，有效状态上集中分配强直接评分对照。
依据[职责决定](../../decisions/D-20261002-contact-risk-constrained-scoring.md)及
[已有工程](P-20261002-contact-risk-guard-engineering-result.md)。

模型SHA73c3e106f9716441f9dccda92b853ddea6eb4b1499724207656d48a6845e76b1，
全部网络/归一化冻结，Cm显式any-contactloss训练与独立direct高度评分。
生成器仍32步/.15/clip1、2pp joint/contact/geometry与5pp support、不确定性
meanGain-1.645std>0、|z|>8回Cup；不改变HF22失败高度门。

独立科学seed651–658，assignment=seed+16000，proposer=assignment+30000。
96env、max300ticks、first episode每early/clear最多2个H10执行窗口；当前joint
连续3步、rest高度>=5mm、warmup10、剩余episode>11、不重置污染。
每次检测/执行后6tick冷却。检测可以重新观察并再规划，不消耗执行窗口配额。
只接纳执行前已知的：unguarded相对Cup joint下降>2pp或any-contactloss上升>2pp，
且Cm与unguarded初始实际PD最大差>1e-5。不得读取未来结果决定接纳。
保存所有检测（包括拒绝）与当前输入/全NN预测/规划，审计资格、介入覆盖及成本。
片段内权重与起始旋转锚固定，六专家每步反馈；外部动作base，不叫完整Cm策略。

5执行臂：base0、Cup1、Cm-guard2、unguarded-direct3、shuffled-guard4。
16slot均匀独立抽样：[0,1,2,2,2,2,2,2,3,3,3,3,3,3,4,4]。
概率.0625/.0625/.375/.375/.125；state-only风险对候选恒定，工程证实其
程序等于unguarded，故实际臂3同时是state-only风险控制器。随机候选5–7
保留预测，不执行、不当真值。所有当前程序与冻结nuisance完成后才分配。

保存冻结Cm全网络3成员×8候选×4指标：支持高度减pre正高度(mm)、H10任一步
接触损失、几何掉落、末3步joint存在。base使用自由旋转law0，其余锚定law1。
实际接触损失为H10任何hand或object force/mg<=.1；joint为末3步共同存在；
支持高度为末3步接触且meshCLR>=2mm的最小正高度减pre正高度；几何掉落为
窗口曾CLR>=2mm后任一步<2mm。force是存在代理，不是识别手物接触对。
初始clear状态掉落另报支持数；H10不是最终45tick稳定抓取及后续掉落验收。

主要估计器固定AIPW mu+I(A=a)/p*(Y-mu)，原HT并列，不依显著性切换。
primary是所有独立接纳窗口的Cm-vs-direct接触损失及高度代价；不以已知分布
难以支持的cal/held分拆充当新训练。仍存SHA12651 group buckets，按group和
完整episode聚类1000次90%bootstrap，种子14753/14754。
best-fixed只用bucket<50按支持高度选base/Cup，其他bucket仅报告该固定臂。
base/Cup/shuffled全样本实际差异及区间并列报告，是后续完整归因输入，不能
因primary通过就宣称所有对照通过或最终Cm策略增益。

支持门：接纳>=64窗口/24episodes/8motion-start组，Cm/direct各>=24窗口/
12episodes/8组；shuffled>=8、base/Cup各>=4窗口。实际Cm至少12窗口在
同一实际轨迹上相对unguarded的PD改变，覆盖>=1%全部检测状态。
primary门：接触loss差<=-5pp，group与episode90上界<0；支持高度差>=-2mm且
两种90%区间下界>=-5mm；几何掉落点差<=+2pp、joint点差>=-2pp。
重复相同direct slots8–10 vs11–13、Cm slots2–4 vs5–7按各3/16校正
残差形成零差噪声；接触风险改善需>=max(5pp,2|direct零差|,2|Cm零差|)。
保存高度零差并与已实际采集的base/Cup参考重复噪声明确区分；不冒充同状态重放。
全部门通过才primary机制PROMISING；支持不足UNCLEAR，否则UNPROMISING。
Probe不提供逐状态regret/oracle或完整闭环策略因果值。

先固定排除seed649做max160tick、每stratum1、全部实际指令/PD/force/mesh/
前后几何/全检测NN/32步/RNG/未来污染/专家与模型冻结审计；首轮native超时335.66秒已保留，没有完整科学结果。按计算等价修正恢复相同
649/160tick/分配/阈值；工程累计上限900秒（含失败），
至少一个实际Cm窗口区别unguarded才能科学运行，否则停止本采集工程。
统计枚举验证任意mu的已知分配期望、mu0=HT、完美mu残差零、重复臂期望零。

全slot<=3600秒/8GiB，准备预留60秒，旧风险工程16.456秒成本并列计入，
新原生工程累计<=900秒（含保留失败），分析与审计预留120秒；来源与完整审计受剩余总预算限制。
科学panel使用3张启动时空闲GPU（优先4/5/6），新工程单GPU；native内部
600/parent640秒，audit内部420/parent450秒；按实际累计总预算停止。
模型计算/仿真GPU，CPU只做标签/统计/文件；旧数据/训练成本单独报告。
任何phase/input/hash/预算失败停止并保留，不补seed或静默延长；无需新外部授权。

计算修正发生在科学数据启动前：unguarded风险不进入高度目标，将35次诊断
risk调用改为3次批量调用，32次高度优化与权重完全相同。排除590/96等价
检查：权重/最终PD/分数/风险差0，risk轨迹最大3.87e-7；1.929→.507秒。
无unsafe的整个检测batch不计算不会执行的Cm/shuffled程序；存在任何unsafe
则保留完整原batch生成，避免优化梯度缩放改变。全部接纳条件/科学seed/门不改。
固定slot3600秒仍含失败和修正费用；只调整内部工程额度，科学阶段不自动延长。

原生r2的649/160tick已完成：370检测/7接纳，native+audit均exit0，全NN/规划/
物理通过，费用317.067秒。但实际分配[2base,0Cup,0Cm,3direct,2shuffled]，缺
Cm分支覆盖，不能以预测PD代替实际执行。仅补同649的60tick工程分支smoke：
第一个执行前接纳窗口确定执行Cm（slot2，真实p=1），其余仍按原RNG；明确
保存forced标记/概率，完整audit验证，排除所有科学分析和训练。工程累计仍
<=900秒，包含首轮343.856秒parent、r2 317.067秒、修正费用10秒和新smoke。
科学651–658所有窗口仍独立均匀slot分配，加载器拒绝任何forced工程记录。
