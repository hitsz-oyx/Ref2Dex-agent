# Ref2Dex Current Research State

Updated: 2026-10-02

本文件是新 agent 的默认入口。运行细节、seed、分数和失败路径只保留在
对应 experiment card；搜索预算和 family 状态在
[`RESEARCH_QUEUE.yaml`](RESEARCH_QUEUE.yaml) 中维护。

## Current decision

2026-10-01 用户调整下一步方向：先检验接触阶段 Cm 的真实动作控制与短期后果排序，
最终成功率仍是任务验收；不把 HF08 间接监督失败当作 Cm 核心思想反证。
本次源码与旧数据审计已完成：HF08 不直接执行候选、评价仅运行 actor；旧 HF02
已做六专家10步干预/20步观察，但每状态只有一个随机候选的真实后果。
下一步先检验候选优势是否超过局部 base/base 噪声，再拟合任务相关动作条件 Cm，
并验证直接执行与重新观察的作用链条。旧数据不能直接提供逐状态真实 regret。
具体区别、最小 Probe、对照与停止条件见
[接触后果机制决定](decisions/D-20261001-contact-consequence-mechanism.md)。
HF09 候选机会 Probe 已完成12个真实 GPU 分支：二次重复的候选优势2.224mm、
base重复噪声0.210mm，数值屏通过，但仅11/32完整近似配对有效，标签 UNCLEAR。
不能把它称为可靠逐状态 oracle；也没有已抬升状态的掉落支持。
随机后果 Probe 已完成1984个实际2+8窗口，含193个已抬升窗口/28次掉落。
非线性 Cm 的留出后果误差低于 state-only/shuffled，但动作选择门 UNPROMISING：
介入42/393，收益估计1.278mm低于base2.546mm。介入窗口只有6个所选动作、
9个base真值，效应区间跨零，不能据此确认负效应；零匹配掉落也不能当零风险。
HF09 slot3/3 已完成：3941窗口/547episodes，622个介入提议随机分为
309Cm/313base；534episodes实际重新观察与决策。局部接触支持抬升增益
+5.106mm，frame-group描述性95%区间[+1.361,+8.850]，局部信号 PROMISING。
但末3步接触下的保留抬升增益+2.901mm区间跨零，已抬升状态掉落11/140Cm
vs4/122base；安全和四控制归因尚未通过，整体 utility 仍 UNCLEAR。
HF09预算3/3完成，不调旧模型/阈值追门。下一步改进物理目标：预测高度/接触
轨迹、显式相对base动作效果，以保留抬升和条件掉落风险选择动作；不进入 PPO。
设计与结果见 [定向介入卡](experiments/probes/P-20261001-targeted-contact-interventions.md)。
HF10 slot1已完成9个物理模型各1000GPU更新（51秒），但joint-contact MAE
.239未通过原.20门；Brier优于constant，概率过度自信，utility仍UNCLEAR。
HF10 slot2已完成3821个实际五推荐器窗口/534episodes，四项不同动作支持充分。
校准物理Cm的保留支持高度vsbase +.413mm，frame90区间[−.152,+.978]；
vsstate-only/shuffled +.279/+.060mm，vs最佳固定cup −.699mm，原收益门UNPROMISING。
有用的新机制信号是已抬升状态释放标签vsbase −5.470pp，描述性frame90
[−8.167,−2.774]pp；vsstate-only/shuffled也降低，但vs固定cup区间跨零。
这支持转向局部防丢失/保留控制的研究判断，不是稳定抓取或最终Cm utility证明。
提议8.846%，105次真实非base Cm匹配，523episodes重新决策；NN/actor/V均冻结。
所有输入/PD执行合同通过，自有进程结束，GPU释放；包括setup共596.62秒/34.85MB。
HF10预算2/2关闭，旧MAE门失败及新收益失败都保留，不扫阈值追抬升。
下一步HF11已固定直接categorical片段策略的学习合同：Cm提供物理未来特征和
明确动作先验，PPO使用实际片段选择概率，评价训练所得策略时继续调用Cm。
HF11已完成matched训练：on/off各207123有效envsteps/840updates，
初始权重及四批episode预算一致。独立stable45tick且无后续drop为17/384vs13/384，
差+1.042pp，四eval seed描述性t95[−.872,+2.956]pp。关键学习门失败：
on5065/off5014评价决策原始argmax相对冻结先验均0变化，故UNPROMISING，
不能把数值差或trained6/96vsprior3/96称为RL收益。预算1/1关闭。
GPU机械审计确认：固定先验log优势3.807，高于学到的最大相对分数1.283/1.612；
网络学到了概率偏好，但贪心执行被固定先验压住。原生reward同时是参考轨迹
imitation乘积，而非45tick保持/防掉落任务目标。下一步把Cm知识放入可学习
动作评分参数，使用共同的接触支持保留reward，先验证学到的策略确实能执行
不同物理动作。HF12已完成1/1：可学习guide+共同保留reward仍未带来
Cm-on学到的执行改变，5130次评价命令相对初始推荐均0变化；off45/5144变化。
On17/384vs off19/384（−.521pp，四eval-seed描述性t95[−7.220,+6.178]pp），
原学习/收益门UNPROMISING，预算关闭，不做lambda/reward/epochs追门。
GPU审计：on guide1.630、最大NN相对分数1.352，推荐仍胜出；参数确实可学习，
不再是旧外部固定先验。候选片段占on控制帧4.952%、非base片段仅.485%；
稀疏控制是下一步待检验的结构假设，不是已证实唯一原因。
两组各207028有效步、四批episode预算/初始权重匹配，完整物理/MC/动作/标签
复算通过，评价冷输入匹配；后三批训练obs差异保留，不把同seed当完整状态匹配。
所有12phase终态/输入hash/冻结合同通过，自有进程退出，GPU释放；含工程34.03分钟/
335.3MB。下一步先设计预测窗口内持续执行的专家/保持片段，实际验证保留机会，
再训练与该新控制合同对应的Cm；不把旧2+8模型冒充10步持续控制预测。
HF13已完成3404实际窗口/573episodes：留出已离桌536状态，hold204/base218
匹配支持充分。固定当前关节姿态hold的保留支持高度−25.191mm，frame90
[−33.629,−17.382]mm；几何失去离桌间隙+21.922pp，fit最佳合格固定候选
仍base，原门UNPROMISING1/1关闭。PD/geometry/propensity全回放0误差，
自有PIDs退出/GPU释放；含工程23.21分钟/37.7MB。不对该候选直接拟合Cm。
已完成原始force单位工程核对：96env×60安静支撑帧，物体净接触力中位
.025529N，约1.003倍实际重量；旧.1N代理在5760帧均不触发。旧false不能
直接解释为真实接触丢失，旧结果继续按强力代理合同保留；HF13几何损失不变。
下一步补测完全悬空负例，给新数据保存原始力/质量及归一化接触代理，再设计
腕部保持/手指继续反馈的候选；不会回填旧bool标签或改旧失败门，C3仍OPEN。
见 [原始力审计](experiments/probes/P-20261002-contact-force-units-r1-audit.json)。
悬空负例亦通过：480帧rawforce全0；新重量归一化代理支撑5760/5760触发、
悬空0/480触发，仅证明两个受控条件，不证明实际抓取接触识别。HF14已完成
3964窗口/584episodes，原收益门UNPROMISING1/1关闭。留出777已离桌状态，
fit选腕部保持/base指反馈，146候选/310base匹配，保留高度−7.985mm，
frame90[−12.842,−4.179]；几何间隙丢失−6.435pp（posthocframe90
[−11.787,−1.504]pp），是保留/继续抬升的取舍，不能算Cm收益。cup指反馈
方案高度−12.387mm且几何损失+4.505pp。posthoc3cm局部阈值保留差
−.644pp，区间宽跨零，不改旧失败门或视为新目标过门。1109腕部保持窗口
手指确实持续变化；专家/rawforce/PD/geometry/propensity复算全0误差；
输入未变，自有PIDs退出/GPU释放，含工程21.66分钟/278.2MB。下一步路线复盘
控制的平移/旋转职责，保留负责抬升的平移反馈，不继续整腕冻结参数扫描。
见 [新候选卡](experiments/probes/P-20261002-wrist-feedback-opportunity.md)。
HF15已按控制职责复盘实现只稳定腕部旋转、保留专家XYZ/手指反馈，正在
GPU工程检查通过：99窗口/47episodes、26旋转稳定程序的平移/手指保持反馈，
全部复算0误差，工程111.24秒/10.14MB。HF15 slot1已完成3917窗口/587episodes，
1409initiallyclear；fit选rotation_cup7、最佳固定也7。留出704clear状态，
134候选/274base匹配，保留支持高度+26.509mm，frame90[19.241,32.794]、
episode90[16.602,36.590]mm；几何丢失−4.972pp，joint-force−1.776pp。
原收益/支持/风险门全部通过，局部候选机会PROMISING，不是Cm或稳定抓取收益。
全12专家/rawforce/PD/XYZ/finger/geometry/propensity/hash检查通过，PIDs退出/
GPU释放，含工程21.28分钟/274.7MB。下一步slot2用新H10数据实际拟合plan-Cm，
与state-only固定程序头、shuffled、base和最佳固定7比较，再做新数据直接控制。
Slot2已完成真实9模型×1000GPU更新（42.81秒）；Cm heldheightRMSE19.85mm，
固定程序物理头17.79、shuffled22.57。校准Cm margin44.565mm，704heldclear
状态0非base提议；修正评分/校准的集成一致性后gain gate仍0，原raw解码器
直接控制UNPROMISING，不降低门槛。固定程序物理头255/704提议，但这只是
预测提议，并非执行收益。该模型通过目录索引区分程序，本身也是程序条件Cm；
已完成6对照模型各1000GPU更新（25.90秒），原目录Cm权重/校准完全复用。
共享冻结选择规则下Cm255/704、局部收益策略279/704、head-shuffled0提议。
新闭环已完成821实际H10窗口/161episodes/25初始组，支持和真实改变门通过。
Cm170窗口中91不同于base、160不同于固定7，662/1700实际控制帧非base。
vsbase保留高度+24.125mm、初始组90[13.704,31.583]；vshead-shuffled+23.538mm。
但vsstate-policy−4.445mm区间跨零，vs固定7−27.651mm90[−34.701,−17.101]；
原全四控制收益门UNPROMISING，HF15预算2/2关闭。所有12,315次模型决策/
PD/专家及参考命令/rawforce/geometry/outcome回放通过，ownPIDs退出/GPU释放；
含全部拟合/失败工程/采集/诊断25.19分钟/97.54MB。后验4105真实H2prefix
高度RMSE8.733优于常速度10.748，但jointBrier.0204差于存在代理延续.0116。
不是完全没物理信息，也没有超过强固定程序；原失败门不改，C3仍OPEN。
HF16 slot1信息门已PROMISING：39170真实step，held已离桌6706/138ep/27组；
Cm高度RMSE4.149 vsstate4.824/shuffled4.850mm，CLRMAE2.271 vs2.855/2.864mm，
jointBrier.002318优于persist.002982。输入pre-state/rawforce/实际nativePD目标
时序、联合height-support标签及fit-only归一化独立复算通过，9nets×1000GPU
更新61.49秒。此为一步物理信息，不和旧两步/H10误差直接比较。
HF16 slot2已终态完成903窗口/175ep/28组，Cm129/179窗口真实改变base、
177改变fixed。IPW联合保留高度vs状态策略+28.284mm90[16.221,40.889]、
vs动作打乱+37.218、vsbase+41.089，但vs强fixed7−1.67190区间跨0，
几何风险+2.769pp未过门；vsbase/shuf净力存在代理也未过原−5pp门。
原全四对照门UNPROMISING，预算2/2关闭；一步物理信息PROMISING保留。
27,090模型决策/实际PD/原始力/geometry/标签独立回放通过，专家与模型
冻结、所有本任务PID退出/GPU释放，累计25.30min/113.68MB。当前较弱base
回退与八程序目录的机会不足是待测假设，不当作已经证实的失败原因。
见 [HF16结果](experiments/probes/P-20261002-native-pd-direct-control-result.md)和
[执行器条件设计](decisions/D-20261002-native-pd-consequence.md)。
见 [对照边界复盘](decisions/D-20261002-catalog-consequence-controls.md)。
不声称HF14已证明旋转是掉落原因。此为最后一轮腕部保持
候选机会Probe；正向才拟合对应新Cm，失败转更高层候选生成，避免关节扫描。
见 [旋转稳定卡](experiments/probes/P-20261002-orientation-feedback-opportunity.md)。
见 [可学习恢复设计](decisions/D-20261001-trainable-recovery-guide.md)。
现有物理NN/六专家未改；C3仍OPEN。
见 [恢复策略设计](decisions/D-20261001-recovery-option-learning.md)。
见 [轨迹模型卡](experiments/probes/P-20261001-contact-trajectory-model.md) 和
[保留支持控制卡](experiments/probes/P-20261001-contact-supported-height-control.md)。
见 [机会结果](experiments/probes/P-20261001-contact-consequence-opportunity-results.json)和
[随机后果排序卡](experiments/probes/P-20261001-contact-consequence-ranking.md)。
HF09 原数据/checkpoint hash未变，自有 GPU4进程已结束，不重开 HF08/HD02。
当前授权以用户最新方向为准；旧 Goal 的“无新 Cm 接法”属于该已完成诊断的边界。
MISSION claim 不变，baseline PARTIAL、Cm utility OPEN；旧结果与输入继续保留。

以下为已完成的 HF08/HD02 处置事实。

当前用户 Goal 已完成：paired evaluator 的完整冷初始状态、RNN、随机数、动作
轨迹和成功后掉落合同通过审计；真实闭环 plain-off 重复性未通过预设门槛。
两次成功为34/384和35/384，63个成功标签、61个掉落标签变化；平均成功率差
仅0.26pp，不能掩盖逐 episode 的不稳定。Wilson95%标签分歧上界为20.44%/19.88%，
超过5%门槛；这不是把标签分歧率当作净成功率噪声或正式统计功效结论。
HD02 已关闭，当前 HF08 实现已停止（KILLED）；条件三臂推理阶段未启动。
原 checkpoint、数据和结果均保留且 hash 未变，无 V/PPO 训练或新增 Cm 接法。
Cm policy utility 仍未证明、C3 OPEN，self-trained baseline PARTIAL。单 GPU4
累计22.02分钟/4.50GiB；自有进程结束，GPU释放。16项针对测试通过。
不得继续本实现的调参、训练或策略实验；Objective A 后续须独立 baseline/curriculum
路线，不能混入 Cm 诊断。
见 [评价器分辨率决策](decisions/D-20261001-paired-evaluator-resolution.md) 和
[HD02 固定实验卡](experiments/probes/P-20261001-paired-evaluator-resolution.md)。

HF08 physical-value Probe 已完成：r7 的 48/48 native 评价完成，终点成功率为
plain_off 41/384、direct_q 40/384、cm_value 33/384；原生 gate 为
`UNPROMISING`。HF08 family 此前为 `PAUSED`，现因 HD02 评价分辨率门失败而
停止当前实现（`KILLED`），Probe 预算 1/1 已用；不升级
Validation，也不继续这条实现的局部调参。North-star scoreboard 不变，Cm policy
utility 仍为 `OPEN`，最终因果解释受实验卡记录的同源重复性审计边界约束。

此前 Mission-level Goal 的任务曾通过固定 Broker 运行；该旧工作流现已退役，当前会话直接
承担后续研究、实现和验证。当前交付期限仍为 2026-10-03 23:59（Asia/Shanghai）；资源边界见
`CAMPAIGN.md`。root 已验收 [HF08 R2 价值目标审计](handoffs/HF08_VALUE_TARGET_AUDIT_R2_20261001.md)：
合同和标签 provenance 通过，主成功为 15/1920、holdout 成功为 1/384；两个 e420
在线 V 均实际完成 7680 次 optimizer update。结论仍为
`UNCLEAR / TRAINING_SUFFICIENCY_OR_DISTRIBUTION_UNKNOWN`，不能写成需要重训或已收敛。
用户随后明确要求 root 直接接管采集器修复及 V 诊断。现已完成：两个冻结 e420
策略共 107584 行、192 个完整首回合，数据与 next-value/主成功/drop 合同通过。
保存 V 对冻结 GAE 诊断目标的 RMSE 为 25.43，原 PPO critic 为 17.17；冻结 GAE
与完整 realized MC 的 RMSE 为 95.35。7 个主成功 episode 均随后掉落；该稀有分层
及单条 realized return 不支持校准、收敛或 bootstrap 偏差结论。当前 factual V 拟合
与目标/回报差异均待区分，不预定增加 V 更新即可修复。完整证据见
[直接接管诊断交接](handoffs/CURRENT_POLICY_VALUE_DIAGNOSTIC_DIRECT_20261001.md)。
采集器已修复 raw logstd/sigma 调用、435-D context 及完整 episode 导出；实际
DExplore 原路径已有外置观测归一化，旧“漏掉归一化”归因已更正。24 项针对测试通过，
所有自有 GPU 进程结束；本诊断无训练，不重置 HF08 的已用收益 Probe slot。

用户随后授权有界 V 额外拟合，HD01 GPU Probe 已完成：一张 GPU、每个 V
1000 次更新、约31秒。相同表示的 fit GAE RMSE 降低75.8%/70.3%，episode-disjoint
holdout 降低18.1%/45.8%；s286 未过预设20%门，联合 gate 为 `UNPROMISING`。
网络能够进一步拟合当前事实目标，但不能据此认定原在线训练不足或已收敛；留出集
完整 realized MC 误差一升一降，主成功覆盖为 fit6/0、holdout0/1，稳定抓取价值的
泛化仍未解决。HD01 预算1/1关闭，不增加更新/换 seed 追门槛，当时 HF08 为 PAUSED；当前处置见 HD02 关闭结果。
结果见 [GPU V 拟合卡](experiments/probes/P-20261001-current-policy-v-fit.md)。
GPU 已释放，源数据与 checkpoint 未修改；GPU 适合本次重复 GRU 训练，历史
CPU-only 阶段限制不构成当前 GPU 禁止。North-star scoreboard 不变。

用户于 2026-09-26 曾授权新的 HF05 goal；该 CPU-only selective causal gate
已完成并判定 `UNPROMISING`。它只在 1/126 个 holdout 状态介入，held-lift
没有超过 always-base，coverage/policy gate 失败。历史 HF01–HF05 仍保持冻结，
但该历史处置不构成对所有后续 GPU 或新 Cm 路线的全局禁止。结果见
[`P-20260926-selective-causal-gate.md`](experiments/probes/P-20260926-selective-causal-gate.md)，
路线处置见 [`D-20260926-after-hf05-selective-gate.md`](decisions/D-20260926-after-hf05-selective-gate.md)。

独立的 C1 六专家初始观测路由已完成五 seed matched Validation。10/10 native arm
及输入/配对合同有效，全部预注册门槛通过；root 在用户委托路线选择后接受该
**任务限定** `SUPPORTED` 结论，并冻结这一路由配置。它不解除上述 Cm 冻结。
见 [C1 决策](decisions/D-20260927-c1-observation-route-validation.md) 与
[Validation 卡](experiments/validations/VAL-20260926-observation-six-expert-c1.md)。

2026-09-27 的后续 Decision Checkpoint 中，用户答复原文仅为 `A`。按当时
Checkpoint 对选项的定义，Option A 表示继续冻结 Cm、保留 C1 substrate、本阶段
使用 0 GPU，并等待一个可预声明且区别于 HF01–HF05 的新高层机制。这是历史路线处置，
不是新增科学证据；阶段性 handoff 仍为 `UNCLEAR/NOT READY`。见
[本次决定](decisions/D-20260927-cm-freeze-option-a.md)。

2026-09-28 的后续授权已移除把 root 停在预设 Option A/Option B 之间的流程依赖。
root 可以在 `MISSION`、`CAMPAIGN`、现有证据和安全边界内自主选择后续路线，并记录
简短 decision memo；本授权不改写上述历史标签，也不把 C1 的任务限定证据升级为更强
的科学结论。用户同时已授权六专家蒸馏与一条新的 Cm 探索路线；这两条路线仍须遵守
`CAMPAIGN`、资源上限、preflight、matched control 和停止条件。r6 support collection
已经完成并通过 root 审计：fit 189 行、holdout 186 行，六个随机 assignment 臂均有
至少 30 行且 split episode 不重叠。第一阶段 CPU student Probe 随后判定
`UNCLEAR`，因为两份数据的 C1 teacher label 全部为 `source_e260`，不能识别六专家
蒸馏。North-star scoreboard 保持不变；scratch Cm CPU support/calibration gate
随后以 `NO_GO` 结束，因为记录中没有可验证的 `object_lift_axis` 及其坐标系
provenance。随后已在主分支修复该合约：新的 evaluator 会按触发时物体四元数将世界
`+Z` 逆旋转到 `object_local_at_trigger_t`，并在每条记录、manifest 和 adapter 中保留
单位轴及其 provenance；旧 r6 数据仍不具备该字段，不能回填或用于拟合。

2026-09-30 root 曾在用户授权范围内并行派发 fit-only CPU 校准修复和独立六专家
轨迹蒸馏；两者预算分别为 2 CPU/15 分钟/1 GiB 与 1 GPU/60 分钟/5 GiB，均不产生
正式 Cm claim。旧 r2 只是 ridge 加 in-sample residual screen，不能据此否定设计
MLP 或整条 Cm 路线；相关历史边界和交接记录继续保留作证据。

## North-star scoreboard

| 目标 | 当前状态 | 证据边界 |
| --- | --- | --- |
| Self-trained grasp | `PARTIAL` | 冻结六专家初始观测路由在限定 12-motion 任务上通过正式 C1 路由/held-lift 门槛；仍不是单一观测驱动 actor 的稳定结果。 |
| Cm one-step information | `PARTIAL` | 随机动作干预中有可学物理效应；信息依赖表示、分布和目标。 |
| Cm policy utility | `OPEN` | 尚无跨训练 seed 的 matched Cm-on > Cm-off 证据；effect-rank 正式 Validation 的正向主张已 `REFUTED`。 |
| Generalization | `OPEN` | 未见物体和多轨迹上的 Cm 收益尚未建立。 |

最终研究价值由第三项决定：在足够可用的 self-trained substrate 上证明 Cm
对真实策略决策有因果增益，而不是只提高离线预测指标。

当前阶段聚焦固定自训练任务分布内的 Cm policy utility；跨物体泛化暂不作为
本阶段门槛。见 [目标重述](decisions/D-20260925-cm-goal-reframe.md)。

## Confirmed long-term facts

- 自训练六专家层级是当前任务内已验证的抓取 substrate。C1 观测臂仅由
  初始 actor observation 选择专家；matched 固定参考臂使用特权物体身份。
  五个 holdout seed 的初始专家选择一致为 311/320、cup 30/30，held-lift
  为观测路由 123/320、固定参考 118/320；预注册联合门槛全过，窄范围
  `SUPPORTED`。这不证明单一 GRAB actor、未见物体泛化、观测路由优于固定
  参考或 Cm utility。详见
  [Validation 卡](experiments/validations/VAL-20260926-observation-six-expert-c1.md)。
- 均匀共享多轨迹 actor 与全池 actor 的近期 Probe 未形成稳定抓取底座；不再
  继续在同一均匀续训方案上堆 epoch。
- 初始动作 option-value、短时/持续接触切换、局部残差和 route-specific
  progress reward 都没有通过预设的 policy-utility 门；不能把单 seed 或离线
  择优结果升级为 Cm 结论。
- 预测接触 pre-contact credit 在单组 gC 上有局部正向信号，但完整 route
  仍为 Cm **14/128** 对 Cm-off **18/128**；该精确变体已停止，不再扫描同一
  系数或门限。
- 现有 matched 结果必须保留 Cm-off 对照；任何新 Cm Probe 都应复用同一专家
  组合，并先做最小可判别实验。
- 2026-09-28 six-expert support collection r6 的 fit/holdout 已通过 canonical
  adapter 和 provenance 审计（189/186 行，六臂覆盖、`1/6` propensity、episode
  disjoint）。其随机 candidate actions 可用于后续 support 检查，但 C1 router
  teacher label 在两 split 均只覆盖 `source_e260`；CPU student Probe 因此只是
  source-only 可预测性证据，不能升级为六专家结论。
- 带 `object_lift_axis` 合约的 r7 fit/holdout 已完成同样的 canonical 审计（188/186 行，
  六臂均至少 30 行，374 个 episode disjoint）。scratch CPU calibration 的 contract
  通过，但 holdout contact q10 和 delta 区间覆盖均未过门槛，结论为
  `UNPROMISING`；详细输入/输出 hash、离线 label 统计和边界见
  [`CM_SCRATCH_CPU_CALIBRATION_R2_AXIS_20260928.md`](handoffs/CM_SCRATCH_CPU_CALIBRATION_R2_AXIS_20260928.md)。
- 允许的一次正确 cwd 工程 smoke 已完成：`agent_temporal_cm_smoke_20260926_r3`
  在 GPU4 上成功加载 temporal 模块、写出 reward 日志并保存 checkpoint（代码
  commit `2d5d0b5`）。它使用旧的五步历史/三条 airplane 输入，只证明 wiring，
  不提供 HF02 的策略或离线预测证据。
- HF02 substrate handoff 已接入 temporal 分支：唯一 canonical route 是
  `src/task/CmResidual/configs/hf02_temporal_canonical_route.json`（SHA256
  `afedfa54c8573096c4d2104d3328efba32b5daf11445323792f45eca19c04d16`），即六个
  self-trained experts、三条 airplane motion、`simulator_object_id` 路由。
  59-motion/十 expert 路线及旧 `3/64` Cm-off 证据不可混用。
- HF02 slot-2 的 canonical offline Probe 已完成并判定 `UNPROMISING`：fit seed254
  有 187 行（六臂最少 31），holdout seed255 有 186 行（六臂最少 30）；两份
  payload 的 route/checkpoint/motion hash、first-episode boundary、start frame、
  action equality、propensity 和 finite checks 均通过。CPU IPW held-lift 中，
  `temporal_cm` 相对 `history_only` 为 `+12.903 pp`，但相对
  `action_shuffled` 为 `-9.677 pp`，未达到双侧 `+5 pp` 门槛；supported-lift
  非回归通过。HF02 已冻结，不启动 online/PPO，也不换 seed、horizon、metric 或
  representation 重扫。完整 hash 与 run manifest 索引见
  [`P-20260926-temporal-expert-credit-results.json`](experiments/probes/P-20260926-temporal-expert-credit-results.json)。
- 59-motion six-expert 上的近期 Cm gate follow-up：stable gate matched seeds251–252
  为 on 8/128、off 5/128，低于预设 +5pp continuation gate；当前 checkpoint
  仍为 policy-utility `UNPROMISING`。见
  [`P-20260926-cm-gate-followup.md`](experiments/probes/P-20260926-cm-gate-followup.md)。
- 单一 airplane motion 的 baseline-owned contact-supported credit audit 未通过
  预设 label/action gate；它与下述 HF03 跨 seed 的 post-action handflow audit
  是不同的数据合同。见
  [`HF03_CONTACT_SUPPORTED_CREDIT_AUDIT_20260926.md`](handoffs/HF03_CONTACT_SUPPORTED_CREDIT_AUDIT_20260926.md)。
- 固定 airplane 接触后的 wrist-z 完整 episode value 路线在二臂和更严格的
  三臂 Probe 中均未过预设门：三臂 held-out Cm-aware 离线策略价值 0.6809，
  state-only 0.6667，差 1.42pp，低于 +5pp 门；没有启动 online Cm-on/off。
  见 [三臂实验卡](experiments/probes/P-20260925-cm-postcontact-three-arm-value.md)。
- HF03 `contact_supported_credit` 的 CPU-only retrospective audit 也已完成并判定
  `UNPROMISING`：复用同一 self-trained `source_e260` airplane substrate 的
  seed246/247 fit（123 行）和 seed248/249 holdout（122 行），只保留首次接触、
  完整五步 followup 和首回合 held-lift 标签。动作后 `next_q`/`next_object_state`
  handflow 相对 action-aware 与 post-handflow-shuffled 对照没有达到预设 5% 的
  held-lift Brier + max-contact-lift RMSE 联合改进门槛（held Brier 分别
  `-2.33%`、`-4.36%`；连续 lift RMSE 分别 `+1.86%`、`+4.85%`）。该记录只有
  `motion_id=0`，不支持多轨迹或跨物体结论；HF03 已冻结，不启动新的 physical
  collection、critic/PPO 或 online Probe。结果索引见
  [`P-20260926-contact-supported-credit-results.json`](experiments/probes/P-20260926-contact-supported-credit-results.json)。
- HF04 trajectory-level-credit 的 CPU screen 也为 `UNPROMISING`：121 行 fit、126 行
  holdout；相对 pre-action，held-lift Brier 仅改善 0.29%，连续 lift RMSE
  恶化 12.95%，未过预设的双侧 5% gate。见
  [HF04 card](experiments/probes/P-20260926-trajectory-credit.md)。
- 已停止的 `agent_temporal_cm_online_probe_20260926_on_s254_e280` 已在 manifest
  中标为 `STOPPED/INVALID_IMPLEMENTATION`：KeyboardInterrupt，最后完成
  `epoch 276/280`。原始 `train.log` 保留，运行不用于任何科学结论，也不消耗
  HF02 slot。
- 历史正式证据与边界见
  [`VAL-20260923-CM-EFFECT-PPO`](experiments/validations/VAL-20260923-CM-EFFECT-PPO.md)
  和相关 Probe cards，不在 STATE 中复制具体运行矩阵。

## Active hypothesis families

| Family | Claim | 状态 | 预算状态 | 分支 |
| --- | --- | --- | --- | --- |
| `HF18` contact-to-lift macro-Cm | `C3` | `UNPROMISING`（early轨迹信息门失败） | slot1后关闭，不启动slot2 | `agent/cm-contact-to-lift-macro` |
| `HF17` strong-reference corrections | `C3` | `UNPROMISING`（固定修正未过留出机会门） | slot1后关闭，不启动slot2 | `agent/cm-strong-reference-corrections` |
| `HF16` native-PD/force-Cm | `C3` | `UNPROMISING`（一步信息正向，强控制门失败） | 2/2关闭 | `agent/cm-native-pd-consequence` |
| `HF15` translation/orientation-plan-Cm | `C3` | `UNPROMISING`（候选机会正向，闭环未过强控制门） | 2/2关闭 | `agent/cm-executable-options` |
| `HF14` wrist-anchored-finger-feedback | `C3` | `CLOSED UNPROMISING` | 1/1；保持/抬升取舍，旧门保留 | `agent/cm-executable-options` |
| `HF13` executable-pose-hold | `C3` | `CLOSED UNPROMISING` | 1/1；固定整姿态失去几何保留 | `agent/cm-executable-options` |
| `HF12` trainable-recovery-guide | `C3` | `CLOSED UNPROMISING` | 1/1；on贪心执行未学到改变 | `agent/cm-trainable-recovery` |
| `HF11` recovery-option-learning | `C3` | `CLOSED UNPROMISING` | 1/1；固定先验压住执行 | `agent/cm-recovery-option-learning` |
| `HF10` physical-trajectory-retention | `C3` | `CLOSED UNPROMISING` | 2/2；局部释放信号，收益未过门 | `agent/cm-contact-trajectory` |
| `HF09` contact-consequence-direct-control | `C3` | `CLOSED UNCLEAR` | 3/3；局部抬升PROMISING，风险未过门 | `agent/cm-contact-consequence` |
| `HF01` local-effect-ranking | `C3` | `KILLED` | 3/3，冻结 | `agent/cm-option-value` |
| `HF02` temporal-cm | `C3` | `PAUSED`（slot-2 UNPROMISING） | 2/3 | `agent/cm-temporal` |
| `HF03` contact-supported-credit | `C3` | `KILLED`（Probe UNPROMISING） | 1/1，CPU gate failed | `agent/cm-contact-credit` |
| `HF04` trajectory-level-credit | `C3` | `KILLED`（Probe UNPROMISING） | 1/1，CPU gate failed | `agent/cm-trajectory-credit` |
| `HF05` selective-causal-intervention | `C3` | `KILLED`（Probe UNPROMISING） | 1/1，CPU gate failed | `agent/cm-selective-causal-gate` |
| `HF06` scratch-offline-teacher-arbitration | `C3` | `KILLED`（teacher-envelope UNPROMISING） | 3/3，MLP coverage/stability gate failed | `agent/cm-scratch-mlp-policy-probe` |
| `HF07` physical-prediction-inference-bottleneck | `C3` | `KILLED`（固定 BC 接法 UNPROMISING） | 1/1，真实策略 matched gate failed | `agent/cm-scratch-mlp-policy-probe` |
| `HF08` physical-value | `C3` | `KILLED`（HD02 闭环重复性门失败，非 Cm 核心假设反证） | 1/1，预算已用 | `agent/cm-physical-value` |

新 Probe 必须登记一个 family、递增 `probe_index_in_family`，并通过
[`RESEARCH_QUEUE.yaml`](RESEARCH_QUEUE.yaml) 的预算门。family 用完预算仍无
信息增益时，必须切换高层假设；换 metric、horizon 或 seed 不会重置预算。

## Next step

当前候选机会已过门：rotation_cup程序留出保留支持高度+26.509mm。真实H10
物理模型拟合也已完成；raw动作张量解码器置信门下0提议，不能进入PPO。
HF15/HF16均预算2/2关闭。HF16一步物理信息PROMISING且实际局部控制
胜过state-policy/base/shuf，但未超过强rotation_cup，原风险门保留。
HF17强参考修正1506窗口已完成：fit选中手指半修正，heldvsreference
−9.227mm90[−23.722,+6.520]，机会门UNPROMISING；slot1后关闭，slot2不启动。
全部实际PD/force/geometry/反馈审计通过，21.58min/110.68MB，本任务PID退出。
HF18已完成4586窗口资格与9模型各1000更新，early held971/144ep/30组、
41抬升正例，原监督支持门通过。Cm高度14.476 vsstate15.173/shuf15.208mm、
CLR5.819 vs5.927/6.078mm，均未过对两控制改善至少5%的原门；形成抬升
Brier与联合高度有信息增益但不替代失败门。UNPROMISING，slot1后关闭。
全部pre/真实PD/控制律/34标签/fit归一化复算0误差，冻结hash/任务终态通过；
GPU采集后当前执行环境CUDA不可用，记录后全部9个匹配拟合统一CPU125.73s，
默认仍GPU。累计11.13min/65.47MB，未启动闭环/PPO。
见 [HF18结果](experiments/probes/P-20261002-contact-to-lift-macro-result.md)。
fit/cal接触作用检查已完成：端点净力不能直接当作周期平均力；cal5190
动态帧Newton速度更新RMSE61.668m/s，常速度.2946；独立float64复算通过。
具体原因未定位，不对力事后缩放。下一步监督实际速度增量或有效非重力
冲量，不称已分离的手物力。相对几何工程亦通过29110个pre-step观测；
下一步把各指动作与物体坐标下的几何绑定，设计可执行候选修正和相应物理
预测合同，而非继续固定目录/门限扫描。
见 [作用链条检查](experiments/probes/P-20261002-contact-impulse-observability-result.md)。
具体下一路线见 [相对几何与动作生成决定](decisions/D-20261002-contact-geometry-action-synthesis.md)。
随机组合采集器/有界工程启动器已实现；离线193状态/1158命令的专家范围、
固定旋转和历史真实40base/20cup PD复现均通过，目标误差0。独立记录审计已实现并拒绝10类synthetic损坏，34输入hash核对，
GPU访问已恢复：8张3090可见，实际CUDA矩阵运算通过；旧阻塞记录保留。
GPU0有他人任务，当前使用空闲GPU1。新采集器原生工程已完成108个H10窗口、
82episodes；1080步实际组合命令/PD/force/mesh标签、全部pre/post相对几何
独立复算通过，专家回放误差0，原生与审计进程均exit0。首次目录检查失败
已修正并保留，全部成本计入工程记录。下一步登记新随机组合监督/信息Probe；
本次仅证明原生工程可用，完整机制/收益与策略学习仍未完成，C3仍OPEN。
见 [源采集工程状态](experiments/probes/P-20261002-contact-geometry-source-engineering.md)。
不对旧目录或阈值继续扫描，不启动完整PPO/最终成功率矩阵。C3仍OPEN。
近似配对的机会信号不能作为可靠反事实真值；继续用已知propensity和独立
新数据测量收益，而非把单状态预测当全候选真值。
排序与覆盖过门后才检验直接重新决策；不依赖长期 V、不启动
完整 PPO 或终点成功率矩阵；模型计算与仿真默认单 GPU，先固定合同与有界预算。
HF08 当前实现和 HD02 诊断已关闭，后续机制是独立路线，旧证据继续保留。
完整 paired evaluator 证据保留在
[HD02 结果索引](experiments/probes/P-20261001-paired-evaluator-resolution-results.json)。
self-trained baseline 继续 PARTIAL；若后续补 Objective A，应另开明确预算、目标和
停止条件的 baseline/curriculum 路线，不盲目堆 epoch，不混入 Cm 诊断。
HF08/HD02 均已用满1/1，当前实现关闭不意味着 Cm 核心假设被否定。

以下保留 HD02 前的历史路线背景；当前授权和处置以上述 HD02 终止结果为准。

HF01–HF04 的实验卡、manifest、结果索引与 Git 提交已完成只读
[closeout audit](handoffs/HF01_HF04_CLOSEOUT_AUDIT_20260926.md)；HF05 的唯一
existing-record screen 已失败固定 policy/safety gate。训练期 Cm 表征的只读
[路线复盘](handoffs/CM_REPRESENTATION_ROUTE_REVIEW_20260926.md)也确认旧 3D/H10
auxiliary 未过升级门。HF01–HF05 与该 representation 路线均保持冻结，不登记新的
representation Probe，也不更换 threshold、seed 或 target。

主代理复核发现 baseline 注册 thread 在冻结决定之后再次发起 GPU 评估；
相关提交暂不合入 `main`。见
[监督审计](handoffs/BASELINE_POSTFREEZE_PROBE_AUDIT_20260926.md)。

C1 的五 seed matched Validation 已完成；冻结已验证的六专家 checkpoint、
初始观测分类器和评估协议，作为当前任务内的自训练层级 substrate。后验
route-vs-downstream 分层显示初始路由不一致仅占 9/320，而路由一致环境中
有 188 个 observation held-lift 失败；这只是描述性证据，不是机制或因果结论。
见 [C1 分层交接](handoffs/C1_ROUTE_FAILURE_PARTITION_20260927.md)。此前的
[Option A](decisions/D-20260927-cm-freeze-option-a.md) 是历史冻结处置，不是当前要求
用户再次插入选择的门槛。六专家蒸馏与新的 Cm scratch 路线均应先记录区别于 HF01–HF05
的高层机制、预算、停止条件和证据边界，再在现有授权内自主选择或请求缺失的资源授权；不得
从该分层或 C1 结果推导 Cm 增益，也不得把历史冻结标签改写成新的实验结果。

r6 support collection 和 source-only CPU distillation 已完成审计。固定 Cm-off
teacher label 的六专家覆盖不足；`agent_cm` 的
`CM-SCRATCH-TA-20260928` CPU-only contract/calibration gate 已按
`UNVERIFIABLE_OBJECT_LIFT_AXIS`、`SCRATCH_CONTRACT_VALIDATION_BLOCKED` 和
`NO_CM_CALIBRATION_AFTER_CONTRACT_STOP` 结束。它确认一步 delta、五步 contact、六臂
assignment、`1/6` propensity 和 episode disjoint 均有效，但没有猜测缺失轴，也没有
把 `source_e260` 当静态 fallback。正式 handoff 见
[`CM_SCRATCH_CPU_CALIBRATION_R1_20260928.md`](handoffs/CM_SCRATCH_CPU_CALIBRATION_R1_20260928.md)。
主分支当前 CPU preflight 已返回 `READY_FOR_COLLECTION`，随后完成了新的带轴 fit/holdout
support collection：fit 188 行、holdout 186 行，六臂均至少 30 行，374 个 episode 全局
不重叠，axis finite/unit 和 provenance 均通过 root 独立复核。`agent_cm` 随后通过了
scratch contract 并完成 CPU calibration，但 holdout contact q10 下界覆盖率只有 0.7688、
delta 区间坐标覆盖率只有 0.2634，两个预设 gate 均失败，校准结论为 `UNPROMISING`。
因此该历史 calibration slot 停止：不从此 artifact 生成可用于蒸馏的正式 Cm-on 标签，
不启动该配方的 online/PPO/Cm 训练；后续新机制以本页最新路线决定和独立 Decision Memo 为准。

## HF08 completed Probe

HF08 physical-value 的 r7 native 执行已经完成 48/48 固定评价。终点每臂 384 个
episode：plain_off 41、direct_q 40、cm_value 33；原生 gate 为 `UNPROMISING`。
因此不启动 Validation，不继续该实现的局部调参，North-star Cm policy utility 保持
`OPEN`。同源 checkpoint 的 e0 重复性审计仍是最终因果解释的边界；完整合同、输入
输出 hash、评价矩阵和运行资源记录见
[HF08 实验卡](experiments/probes/P-20260930-cm-physical-value.md)及其结果索引。

R2 价值目标审计已由 root 验收：主成功 15/1920、holdout 1/384，两个 e420 V
均有 7680 次 optimizer update；这些是已验收工程/标签事实，不构成 V 充分性、当前
策略校准、策略效用或收敛结论。后续工程准备不改变 HF08 已用预算。
