# Ref2Dex 当前研究状态

## 2026-10-08 用户修改审查与当前 blocker

consequence-evaluator 用户新增的H匹配、最小唯一配对覆盖和重放状态合同保留。
修复repeat验收遗漏未来手/done、生产门槛误封只读审计、native物理枚举/dtype
序列化，以及pickle对象共享导致内容哈希在save/load后变化的问题。
六角色airplane/duck/cup资格36/8/63（各64条），mixed12/train5/balanced5为0/5/4；
生产route仍不可训练。216条旧连续episode经H匹配重新审计，唯一pair=1/1/1，
低于8/4/4；没有启动evaluator fit。独立native twin r4（a7d5146/GPU0）通过：
三fresh进程同前缀状态/RNG精确一致、repeat全部future/action/done/native状态一致，
请求/实际control L2分别0.75865/1.14996。它是engineering_only且不可训练，
不改变ref2 continuous schema或科学结论。用户明确保留连续rollout、不fork；
twin只作工程诊断。对全部可比较窗口穷举后，物理匹配候选4/1/3、加H仍为1/1/1，
配对选择器未漏掉合格episode pair。下一步只做不可训练的连续保持阶段定向诊断，
核对合格飞机专家的局部反例和匹配覆盖，不放宽生产资格或状态阈值。
124项Task测试及verify通过；原始失败run、用户其他修改均保留。
GPU1/2的三源PointWorld继续原冻结配方与50000步/10:00绝对截止预算。
详见[配对覆盖卡](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-consequence-pair-coverage.md)。

## 2026-10-07 ref2/ref3 运行记录（历史）

新随机自训练s1 parent200资格50/64；s3 bounded220→260固定资格36/64，
恢复Probe PROMISING。十条剩余reference全部恢复；六角色airplane_base已过资格，duck340固定资格8/64，保留旧duck280的0/64记录；cup正训练，其余三角色待训练，
尚无真实evaluator数据/fit。GPU0已自然空闲，duck首轮smoke因缺mesh在PPO前失败；
owned资产补齐；duck真实1024帧geometry smoke通过，533forceproxy帧均有<=1cm gap。
短20epoch不足以复现历史80epoch specialist预算；GPU0重新空闲后，
1386d51从合格s3260完成新duck260→340，端点资格8/64；cup80epoch独立迁移已启动，完成后固定资格已排队；三角色待恢复。ref2改用执行前已知24×18
请求残差计划，三臂E0/object/EI，同容量、几何接触约束、同expert/motion/current
物体手状态配对；89工程测试通过（含motion软链、阶段覆盖、native driver与新路由回归）；GPU1真实8env×128步几何smoke通过，712forceproxy帧均有≤1cm采样gap，
但尚未验证其他物体/桌面反例或完整真实数据，不升级为精确contact GT。

按用户ref3，四源PointWorld正常保存停止step14250，moving-anchor h24验证
Oak9.24/GRAB38.86/ARCTIC51.24/ContactPose27.15mm；best/latest保留。
三源Oak/GRAB/ARCTIC主监督已于23:20启动，commit9019fd4/GPU1/2每卡64，
latest14250模型初始化/fresh优化器，50000新更新，
用户指定截止2026-10-08 10:00。新step0同panel EPE为Oak15.92/GRAB36.13/ARCTIC51.39mm，
不同于旧四源panel，不直接横比。首次250val宏34.48→37.16mm，早期退化，
step1500宏29.88mm曾改善，但step2500宏35.87mm再次高于初始化34.48；
短期波动，不认定稳定改善或平台；
每250步保存latest/best；含val/save实测.744s/update，50000预计09:41，硬截止不变。
ContactPose仅刚性运输辅助；EPIC仍candidate_only，
独立修订坐标约定、双手有效性、gap及OF overlap质量，未开放训练。
旧四源结果不升级为同质监督或Cm收益结论，最终matched trained-policy utility仍OPEN。

## 2026-10-07 用户指定 consequence-evaluator 路线

用户要求将当前PointWorld与数据处理代码合并到本地main，再创建`consequence-evaluator`。
数据扩充提交a620375已合并，实验索引跟进9c555d1；当前新分支继承oracle与PointWorld。
按Task ref1固定K24/Kexec8，先检验连续六专家rollout上的E0(H,A)与Eoracle(H,A,Z_GT)，
暂不推进WM适配或在线proposal。论文原文/源码已核对；独立scalar BT是Robometer改编。
窗口合同、标签mask、两臂模型、有界训练入口与六专家连续采集驱动已实现；CPU测试覆盖真实驱动循环，
但未在真实Isaac Gym上验证，也未采集或拟合真实数据。采集副产物保留measured q/root与扰动审计，标签仍需核验。
独立test评价入口已实现：冻结val选择的权重，报告配对排序与任务/阶段汇总、跨episode未来替换诊断；
31项工程测试通过，不代表真实oracle headroom，也不把重叠窗口当独立样本。
旧oracle outputs、六专家权重未找到，原motion输出缺失。用户已授权重训专家并重新rollout；
外部canonical几何motion与仿真资产可用，13条motion完成CPU合同检查，已复制/软链接到owned输出。
DExplore指向已删除baseline的失效data链接已保留并修复。先新训s1 parent，再检查frame0抓取后迁移s3与其余专家；
新权重不继承旧六专家Validation结论。PointWorld原三卡训练已正常完成10000新更新，源码hash未变；
固定balanced运动anchor h24 EPE14.0672→11.1342mm，best为step9500的11.1026mm，
预训练Probe PROMISING；末尾2500更新仍改善3.49%，仅能判断收益减慢，不能认定完全平台。
best/latest/final均保留，未开启test或新训练预算。专家重建r2的native smoke在首次PPO更新前
因NumPy别名兼容导入顺序失败，原日志保留；调整bootstrap导入顺序后重试原预算内预检。
随后用户授权OakInk2继续：GPU1/2、每卡64/global128，从原latest保留AdamW动量，
沿用末尾lr约1e-5追加最多10000更新，仍受原绝对deadline约束；这是显式两卡迁移，非逐位resume。
GPU0用于consequence专家/rollout。r3重建smoke在首次reset发现CPU/CUDA混用，未发生PPO更新；
task-local入口改用CUDA PhysX tensor pipeline后，按原输入/预算执行r4预检。
启动前用户暂停以上续训安排，优先混合OakInk2/GRAB/ARCTIC/ContactPose，原latest仅作模型初始化，
新优化器/训练从头开始。两卡续训与r4预检均未启动；当前无本会话GPU进程需要停止。
四源全量整合和双卡12步工程检查已完成；GRAB/ARCTIC官方split、原生类别映射，
ContactPose真实世界位姿/30Hz及gap屏蔽已接入，参与者train/val/test隔离。
2026-10-07 19:57:29开始混合训练，运行提交384f860，GPU1/2每卡64/global128，
OakInk2/GRAB/ARCTIC/ContactPose采样50/20/20/10；原latest只加载模型，优化器/调度重置。
训练窗口总4794229，固定64val窗口/source，step500运动anchor h24宏平均52.56→45.74mm；
OakInk2新固定panel9.79→10.49mm，较step250略恢复，其他三源改善，早期遗忘/适配取舍仍UNCLEAR。
沿用原deadline2026-10-08 09:58:37，最多40000新更新，不追加24h。用户随后恢复consequence-evaluator：GPU0启动r4重建，2epoch GPU smoke通过，
64env/200epoch s1 scratch母策略已完成（1114.91s），权重保留；最近接触/保持指标仍0。
固定64条frame0/45帧保持无后续drop资格检查完成（c0b10fb）：0/64合格，
不扩展六专家，先同初始状态参考动作回放诊断；这不构成evaluator/world-model负结论。新增局部物理事件标签、独立专家progress抽样和资格检查共44项工程测试通过；
尚未获得真实evaluator数据或oracle headroom结论。
后续真实物理诊断确认r4存在GPU reset提交/刷新缺陷：首步物体跳1.48m。
task-local修复后同环境降到1.43mm，重复子集reset/FK物理检查通过，51项工程测试通过；
原r4训练和0/64不能用于判断正确实现的学习效果。已找回原s1 corrected tensor，
与r4 canonical在q/坐标/contact上并不等价；旧新均用graspenv+Dexplore_Inspire。
原s1数据GPU物理预检已通过（15.78s，首步位移1.43mm，三次子集reset通过）；
原CmLite权重/转移与s3 corrected输入仍需重建。
当前推进：567c909启动原s1输入/修复reset/Cm-off新重建，GPU0、64env/h32/mb256、
LR2e-5/mini-epochs6、anneal40→80、200epoch；smoke36.18s通过，前80epoch有真实接触/抬升奖励，
训练吞吐约500FPS、显存18251MiB。缺失CmLite奖励未启用，不称原配方精确复现。
CPU已恢复s3 corrected reference（23.17s，几何相对误差<3e-7m，左contact0），迁移仍待母策略资格。
该新母策略200epoch已完成（1123.50s）；独立64条frame0完整episode中50条通过45帧保持/无后续drop，
超过预设8/64数据准备门槛，重建Probe PROMISING（不继承旧Validation或证明Cm收益）。
GPU0下一步从新自训练parent200迁移s3到220，先2epoch工程检查、再20epoch正式Probe部分，
最后固定端点资格检查；权重祖先/RMS/优化器/epoch均核对，LR显式1e-5，保留已结束anneal40→80。
s3首轮200→220已完成，但固定64frame0资格只有2/64通过（未达8），首步位移1.72mm内正常，
有16/64短抬升且训练接触上升。下一步同输入/配方有界续训220→260并再评固定端点；不扩展其他专家。
cup参考输入CPU恢复通过（934帧/相对误差<3e-7m），不是cup抓取证据。
其他专家和连续六专家数据仍未完成；尚未训练E0/Eoracle。
详见[混合预训练卡](../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-pointworld-multisource.md)。
见[重建Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261007-consequence-baseline-rebuild.md)。
见[任务入口](../src/task/consequence-evaluator/README.md)。不改变最终matched Cm-on/off策略utility要求。

### consequence-evaluator 2026-10-08 续跑

六个新 endpoint 已冻结为不同 hash：airplane_base36/64、duck8/64、cup63/64
通过固定资格门；mixed12/train5/balanced5 为0/5/4，保留为 observational
候选。route.json 明确 `all_experts_operationally_qualified=false` 和
`training_allowed=false`，不恢复旧六专家 Validation 结论。queue smoke 的
hard-object oversampling 与 minibatch divisibility 修复已提交
`3b4305b`、`e62dd08`。

首轮三 split 原始采集 144 episodes 后，严格 local-state label 为
train/val/test=0/0/2；独立 proxy/gap 几何审计通过。按 Decision Note 追加
296/297/298 三波次、216 episodes，仍使用 24 步 decision-known residual 和
2 cm hand RMS。追加标签在未改合同下变为 `READY`、pair=2/1/5，审计为
149,841 valid frames、49,388 native proxy、45,984 proxy-near、仅1 proxy-far。
但 2 train/1 val 不足以支撑 32-pair/update 的 evaluator fit；继续训练只会
重复采样并在一个 validation pair 上选模，故本轮不启动 `prepare_windows` 或
`train_matched`，也不放宽阈值。原始、label 和 coverage diagnostic 均保留；
下一步若继续，需设计同一 current state 的 twin residual branches，而不是
再堆 generic waves 或把 hand RMS 放宽后重判。North-star matched Cm-on/off
policy utility 仍 OPEN。

当前实现审查把这项 gate 固化为规则
`geometry-corroborated-H-matched-state-residual-events-v3`：偏好端点必须同时
匹配历史 `H`、当前物体位姿和11点手状态，历史相对RMS上限为0.25；train/val/test
还必须分别拥有至少8/4/4个不同的无序 episode pair。collector 只把该要求冻结进
source manifest；labeler、window 准备、`Windows` 和 `train_matched.py` 均拒绝不满足
的生产 fit 数据。上面的历史
2/1/5 产物因此只保留为 observational audit，不能作为 evaluator fit 输入；当前
route 仍是 `all_experts_operationally_qualified=false`、`training_allowed=false`。

新增的 `consequence_evaluator.twin` 是严格的 Isaac-free v1 合同测试：要求完整
native task/controller buffer、Python/NumPy/Torch RNG、fresh simulator prefix
replay provenance、刚体物体位姿和双分支第0帧锚点。它尚未连接 native collector，
也没有真实 twin branch 产物；这部分仍是下一项实现 blocker，不能把合同测试升级为
科学证据。Task tests 在本轮代码审查后继续作为工程回归，不改变 North-star claim。

2026-10-08 native twin 工程推进：`consequence_evaluator.twin` 新增不依赖 Isaac 导入的
native capture adapter、完整 prefix trace hash/帧数检查和 Python/NumPy/Torch RNG 捕获。
它现在能把初始化后的 native task/controller 边界转成 v1 snapshot，但尚未接入连续
collector 的 fresh-simulator 双分支执行，也没有真实 twin branch 产物；因此 blocker
从“缺少 adapter”收窄为“缺少 native branch runner 与真实覆盖”，不升级为科学证据。

更新：2026-10-06。本摘要整合已交付的主分支与本轮Cm研究事实，不产生正式科研结论，
不纳入其他独立会话尚未交付的结果。完整旧摘要见[状态快照](archive/research/STATE-20260930-before-workflow-simplification.md)。

| North-star | 当前判断 |
| --- | --- |
| Self-trained grasp | PARTIAL：限定 12-motion 的六专家初始观测路由已通过正式 C1 Validation；单一 actor 稳定结果仍未证。 |
| Cm one-step information | PARTIAL：物理效应可学，但依赖表示、分布与目标。 |
| Cm policy utility | OPEN：尚无跨训练 seed 的 matched Cm-on 优于 Cm-off 证据；effect-rank 正式 Validation 的正向主张已 REFUTED。 |
| Generalization | OPEN：未见物体/多轨迹上的 Cm 收益尚未建立，本阶段先聚焦固定任务分布。 |

## 关键事实与边界

- C1 的任务限定 SUPPORTED 不证明单一 actor、未见物体泛化或 Cm utility；冻结已验收的
  六专家 substrate。[正式验证](experiments/validations/VAL-20260926-observation-six-expert-c1.md)。
- HF01–HF05 的已失败局部路线保持冻结，不靠换 seed/门槛重置预算；这不是所有未来 Cm/GPU
  路线的全局禁止。历史路线边界保存在归档实验记录中。
- 用户已授权边界内自主选路线，以及六专家蒸馏与新的 Cm 探索；历史标签不因此升级。
- r6 support 的 teacher label 仅覆盖 source_e260，不能证明六专家蒸馏；其缺失轴字段不能回填。
- r7 轴合约通过，但 contact q10 与 delta 覆盖未过 calibration gate，不生成正式 Cm-on 标签。
  [校准证据](archive/2026-10-04-research-governance/handoffs/CM_SCRATCH_CPU_CALIBRATION_R2_AXIS_20260928.md)。

## 其他已有任务交付边界

主分支已记录的两个旧系统受控任务：一次 fit-only CPU 校准修复，以及独立六专家逐步轨迹蒸馏。
校准任务 2 CPU/15 分钟/1 GiB，蒸馏任务 1 GPU/60 分钟/5 GiB；均只形成 Probe 结论。
旧系统任务的实际终态以各自交付为准，不由新工作流猜测或重新启动。

先验收实际交付，停止失败的局部 calibration tuning；蒸馏独立推进。新 Cm 路线须服务于
真实策略因果增益，保留 matched Cm-off 对照。[实验索引](experiments/INDEX.md)按需检索。
运行细节、失效执行、数值和哈希留在原卡/manifest；资源授权见 [CAMPAIGN](CAMPAIGN.md)。


## 本轮 Cm 策略价值研究

用户授权先取得真实策略 matched Probe，正向则优先正式 Validation；期限为
2026-10-03 23:59（Asia/Shanghai）。完整交互与短期物理预测结合长期价值，
最终要求训练所得actor受益；瞬时物体位移不替代动作价值，跨数据集预训练可选。

HF06 teacher-envelope与HF07 BC物理预测输入实现均为UNPROMISING，保持关闭。
HF08 physical-value已完成：公共池1041599转移、1920完整episode，fit831811行，
开发holdout209788行；同一自训练source_e260、三条canonical airplane motion，
六臂均追加160epoch/327680交互，全部48点actor-only评价完成。原生gate为
UNPROMISING：终点Cm33/384，普通PPO41/384，直接Q40/384。这里只是Probe，
不否定Cm核心假设；North-star policy utility仍OPEN。

R2价值目标审计已由root验收：source主成功15/1920、holdout成功1/384，两个e420
V checkpoint均实际完成7680 optimizer updates。结论为UNCLEAR：V训练充分性和当前
策略校准仍未知；不据此宣称需要重训或已收敛。

在线适配后物体位置误差降至0.78/0.86cm，仍弱于恒定线速度基线；姿态误差
仍约1.34/1.36rad，明显弱于保持状态。模型不能称为已准确预测物理转移。
同source checkpoint的e0重复评价有差异，原生配对只覆盖env/motion/start/时长/
初始高度；初始完整state恢复审计此前已排除为当前阻塞，不以单次realized MC误差
宣称bias。暂不升级Validation，不调参重扫HF08。

collector与V诊断工程已验收；R1 nativecwd资产失败、R2 wrapper在GPU ownership前失败，
两者均无新数据、无V充分性结论。R3/r3b也已FAILED并完成进程清理，均0行；
CM真实输入守卫已验收、集成，main复验11项通过。
当前执行原生Gaussian、frame-0、冻结策略的完整episode诊断，尚无完整采集数据或新训练。
真实环境初始化暴露reward_shaper对象检查错误；修复后首个动作又因采集器误把logstd
当sigma退出。原生model还负责观测归一化，raw网络直调不能替代它。错误属于采集器，
不证明checkpoint标准差无效。root已派固定RL角色做CPU-only原生player合约修复；
原恢复轮截止不延长，未来采集按实际累计成本另记有界预算。单次MC误差不单独证明bias；
HF08 slot不重置。见[修复记录](archive/2026-10-04-research-governance/decisions/D-20261001-native-player-collector-repair.md)。
[HF08实验卡](experiments/probes/P-20260930-cm-physical-value.md)与
[完整结果](experiments/probes/P-20260930-cm-physical-value-results.json)保留边界与数值。
原始数据/checkpoint留在原研究工作树的research/output/P-20260930-cm-physical-value/r7，
未提交Git，不因本次合并移动或删除。

## Ref5：point-flow E → G 与 surface I

当前实现分支 `agent/cm-interaction-oracle`，新 Task 为
`src/task/cm-interaction-oracle/`。新工具/卡已放入 Task；旧工具和历史卡逐步迁移，
根级 Mission、Campaign、seed ledger 与状态保持唯一来源。

- 真正冻结 point-flow K1 E → G 已跑 matched Probe：同容量 H MAE23.3652、
  H+GT E23.2837、GT-trained bridge换预测E23.6477、预测E训练bridge22.8961。
  判定 UNCLEAR：GT仅0.35%改善，预测fit2.01%且CI跨零，收益受单个高RTG episode
  支配。去此episode剩余21个episode预测fit反而差5.82%。
- 同容量GT合同检查（短序列右对齐，避免GRU补零冲淡）：pose K1/K4/K8改善
  0.034%/1.067%/2.748%，full13 K8改善7.42%
  但去高RTG episode后负1.66%；没有合同通过预设门槛。保留K8 pose弱方向性信号，
  不据此扩大K4 predictor或进入policy teacher。
- 8patch×5D GT surface I比同容量零I改善10.83%，CI跨零且约97%收益来自同一
  高RTG episode；pooled I更好，未证明空间topology独立价值，不启动I head/K4 I。
- 工程审计发现fresh shards手root平移最高22.33mm，旧identity-root假设不能迁移。
  用当前measured body pose恢复共同root后，五body最大残差5.35微米；当前/未来root
  cache分开。首轮G失效、no-grad技术失败均留痕；修复后独立review未见其他致命bug。
  CM平移RMSE19.33mm，当前twist persistence4.92mm，不能称准确物理预测。
- 原始episode summary与assembled labels逐条一致：112 episodes仅2次达到45步
  held成功，两次随后都掉落；该source_e260数据的成功无后续掉落数为0。主导held-out
  RTG episode即使最长保持8.87s也随后drop，高RTG不能直接视为最终稳定抓取。
  这些统计不重判已验收六专家baseline，不据此直接训练稀疏success/drop二分类G。

Root选择：保留E物理主路线与修正的几何合同，暂缓局部I/predictor扩展；下一次
最便宜的决策应先核对持续hold/drop任务目标与action contrast。RTG回归不替代抓取、
保持、掉落评价，更不替代matched Cm-on/off训练所得策略的因果收益。
两路使用GPU6/7，均已结束、零新采集；本轮只给Probe判断，North-star Cm utility仍OPEN。

[E→G实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-pointflow-g.md)，
[surface I实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-ref5-surface-i-gt-value.md)。

## Task ref1：RECAP-style 相对任务 advantage

同一 Task/branch 已实现并执行 cross-fitted MC V → 32步 advantage 的有界
Decision Probe；task reward 固定 `held/10 + lift_progress/5 + stable`，排除
base/approach，所有模型显式控制 episode-fixed noise。历史90train/22test
episode split、两次V拟合×三fold，gamma0.99，固定16epochs，不改变核心 Mission。

- G0标签计数、episode支持、集中度均达标；held-out V MSE229.58低于zero324.80。
  但独立fit标签一致率 train34.72%/test39.15%，远低于预设75%，结论UNCLEAR。
  按门槛停止，G1 action critic、G2 predicted E/I、G3 selector均未执行。
- 连续advantage test相关性0.978，去最大MC episode仍0.952，保留方向性线索。
  真实32步task reward非零只覆盖16.3% test queries；该区标签一致78.0%，
  零reward区29.6%。后者fit差中位数0.255大于0.05幅度floor，不只是微小量化抖动。
  不能事后用未来reward active筛选测试样本或改一致率分母宣布通过。
- 工程review修复neutral样本污染binary AUC的问题（正式运行前，G0不受影响）；
  后续112 episode target/row/fold/threshold audit精确一致，无进一步致命问题。
  当前暂停原因是标签/value合同可靠性不足，不能写成RECAP或动作信息被否定。

Root Decision：保留相对任务后果方向，下一决策前提是可信标签/value；暂不投入
I/G容量扩展。任何改标签协议须重新明确假设和判定，不放松本轮门槛或追加seed搜索。
本轮GPU6 model执行含smoke<20s，零新采集，任务进程均结束；代码commit9dd503d。
North-star Cm训练策略因果utility仍OPEN，不把离线标签工作算作policy进步。

[相对任务advantage实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-recap-relative-action.md)。

## Task ref2：冻结连续 advantage 的 action ranking

ref2 明确授权新排序问题，未重跑V或放松旧三分类G0门槛。新 matched pairwise
Probe完整执行 H / HaK1当前动作 / HaK4已记录feedback；同96,705参数、同init、
同pairs/训练设置，既有90/22episode split、全部冻结advantage继续使用。

- 正式测试 macro同motion/phase/noise pair accuracy：H59.712%、HaK1 59.617%、
  K4 60.520%。K1增益−0.095pp；macroSpearman仅+0.0021；打乱当前动作仅降低
  0.133pp，两个独立V target的增益均负；8/19可配对episode改善。UNPROMISING
  针对当前fixed target/data/fit合同，不是物理动作信息或Cm被否定。
- 新审计纠正ref2的排序前提：Pearson0.978并不证明rank稳。两套冻结advantage
  的Spearman train0.363/test0.540，条件pair顺序一致约62.4%/66.8%。去三分类
  boundary并未形成强排序监督；训练93%–95%而测试≈60%，泛化仍弱。
- 配对实际覆盖19/22test episodes、64strata（2,432/2,816anchor queries）；
  macroSpearman使用88strata，包含单episode无cross-pair组，分母明确分开。
  去最大MC episode后K1+1.333pp是弱敏感性线索，未过整体门槛。
- 当前动作进入网络且打乱score RMS0.991；独立review及root SciPy/rawscore
  复算确认pair/input/normalization/target/hash/metrics正确，无进一步致命bug。
  第一smoke仅path-hash metadata失败，修复留痕；正式执行commit1a64eab。

Root按ref2负分支停止这份离线标签上的critic/predicted E/I扩展，未新采集或训练
teacher/policy。下一研究决策应先建立可信物理action/outcome contrast，不能仅
换V seed、width或threshold继续此数据拟合。保留E/I与relative action科学问题，
North-star causal Cm-on/off训练所得策略utility仍OPEN。
本轮GPU6 valid模型执行含smoke<49s，产物<8MB，全部任务结束。

[冻结相对advantage排序实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-relative-action-ranking.md)。

## Task ref3：真实随机动作干预

固定自训练source_e260、airplane三motion，移除episode噪声，七臂随机分配
四步feedback residual；336完整episode、320实际干预、320完整32步窗口。
每臂37–54次、有效dose100%；全batch reset屏障，pre/post/PD/GT标签经独立review。
全部decision为pre-lift（最高lift8.74mm），early-hold/drop-risk样本为0。

- 按环境留出66train/18test，预测器及scorerOOF也隔离环境。H E/I MSE0.5748，
  Ha0.5841；macro任务排序 H66.78%、direct66.14%、predicted65.17%、GT61.17%。
  七项预设gate全部失败，当前合同UNPROMISING，不启动selector/teacher/policy。
- GT保留contact-retention局部信息，但整体泛化弱；模型容量/OOF噪声差和少量
  history瞬态限制归因。PCA总test能量异常主要由单行支配，典型样本保留率中位
  86.96%，不是所有测试状态丢失93%；不删除outlier重判结果。
- 最便宜的既有数据Decision诊断控制当前状态后，物体短期旋转有弱随机臂响应
  （E/I家族探索性permutation tail0.045），任务窗口/完整episode证据仍不足。
  手实际收到不同扰动不等于已建立任务相关可预测中介链。
- Full summaries105/336达到45步hold、76随后drop；不同采样下的raw arm均值
  不当作策略增益，也不修改旧baseline结论。motion symlink漏预先hash已补明确
  post-run provenance，后续collector修复；原manifest/失败smoke保持可追溯。

Root停止本pre-lift E/I8→Y16/32 PCA/MLP扩展，保留随机干预数据和物理control
问题；不据此终止Cm核心路线。未来研究需要新的阶段/时域决策合同，不换seed/
阈值/encoder继续拟合本合同。单GPU6，正式仿真222s、smoke70s、模型4.86s，
统计1.12s，产物<13MB，进程均结束。North-star训练所得Cm策略utility仍OPEN。

[真实随机动作干预实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-randomized-action-intervention.md)。

## Task ref4：early-hold interaction retention control

按用户新增ref4，干预移到lift≥3cm且连续6步contact-proxy的early hold。
1,008完整episodes产生494随机干预，全部32步窗口完整、剂量实际执行。
后续214次失败中212有物理高度损失；98次早失败全部保留，不筛掉step8失败者。

- A UNPROMISING：短接触比例调整后臂间范围5.68pp，tail0.1335；I14 family
  tail0.5005，未建立预设action→retention-I可控链。
- B UNCLEAR：GT I令后9..32预测误差改善49.6%，failure AUC0.899→0.949；
  但两主目标均只有2充分支持分层，未达预设3层。原macro排序58.46→77.33%
  受1–2pair小层放大；充分支持层描述性87.76→96.32%，不事后改gate。
  未早失败的92个test trial也有预后改善，但这只是post-treatment描述性子集。
- 独立工程review及root逐项复算通过。未启动C、selector或policy训练。

Root关闭当前early-hold四步feedback residual→I8合同的继续拟合，保留GT I预后
价值。缺口转为可操纵的保持控制变量/动作时域，不能靠更多seed、网络或补B支持
绕过A失败；不否定Cm核心假设。单GPU6仿真约417s、模型2.31s、产物<18MiB，
全部结束。North-star训练所得matched Cm-on/off策略utility仍OPEN。
[early-hold实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-early-hold-intervention.md)。

## Task ref5：延长 feedback residual 的物理响应

仅做physics Decision，不训练Cm/S/PPO：在同一early-hold区随机分配7方向×K4/8/16，
2,016完整episodes、1,006完整干预窗口。非零cell最少32、两半最少13、pooled zero148；
全部剂量约1.0，native PD和独立标签/OLS/curve复算通过。

结果UNPROMISING：短接触tail0.1645，K16最大绝对效应6.62pp，未达10pp；I16 family
tail0.572，后17..32接触/高度失败tail0.581/0.244，未形成稳定duration→retention-I链。
动作实际改变了手：wristx+在同一step16调整位移随K为4.51/10.47/21.92mm。
部分反向baseline命令与补偿一致，但累计剂量/释放恢复时长同时变化，不能唯一识别
feedback cancellation。secondary all32高度失败有弱tail0.085，不能救援I控制gate。

Root关闭当前K≤16 feedback-residual合同，保留I预后信息及测得的反馈响应；不扩大网络、
时长/幅度/seed扫描或补旧B支持。不否定其他operator、I/Cm核心思想；固定绝对target
仍未测试，若以后推进须有能区分机制的新合同。GPU6仿真787s、CPU统计2.81s，产物
<37MiB，全部结束；25 Task tests与变更检查通过。训练所得matched Cm策略utility仍OPEN。
[duration实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-early-hold-duration.md)。

## Task ref6：固定 K8 的幅值与局部 interaction authority

alpha1/2/4×七方向联合随机：1,680 episodes、863完整干预窗口，剂量、support、
原生PD、输入哈希及独立/root复算全部通过。任务关联authority gate UNPROMISING：
短接触tail0.8895、alpha4最大效应3.68pp；后9..32接触/高度失败tail0.554/0.1965，
没有合格阈值候选，因此未启动条件性Stage2或神经模型。

局部interaction响应PROMISING：I8、切向投影、ratio family tail均0.0005，主要由
thumb_distal贡献；alpha4 finger−距离+22.95mm且力下降，finger+距离−8.44mm且力上升，
两半同向。wristx+同step8手位移12.95→23.53→47.61mm确认动作authority。
原force norm已捕捉信号，不能将缺少任务差异全部归因方向压缩；全手OR contact-proxy
可掩盖单thumb变化，物理高度结果也未明确分化。只证明此合同的局部proxy可控，
未识别抓持阈值、可用控制区间或Cm收益。

Root保留局部action→I证据，停止本次幅值合同，不扩大幅值/seed/网络以重复authority；
后续需区分可控局部接触是否实际参与承载。GPU6采集player累计671.48s，产物<40MiB，
均已结束；North-star Cm utility仍OPEN。
[幅值实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-amplitude-authority.md)。

## Task ref7：独立 finger DOF 与实际幅度审计

拆为六独立DOF±（含thumb-yaw）、zero、匹配旧synergy±共15臂，固定K8/20%driver range。
1,680episodes、854完整窗口，support/dose/nativePD/17input hashes及独立复算通过；
各指PD rad/deg、mimic、实际q和真实tip矩阵已写入Task实验卡，ref6旧卡也补PD幅度表。
同20%range并非同毫米：四指局部tip对照范数约15–29mm，thumb约9–14mm，未假定thumb
运动最大或已知冗余/承载身份。

预设own-I+late-task及useful gates均UNPROMISING；但后9..32contact family tail0.0005，
不能概括单指控制无效。middle−保持−10.94pp两半重复；thumb-yaw−保持+10.76pp、高度
失败−11.34pp，两半同向。后者height family tail0.1535及own-I幅度/半包一致性未达标，
保留为下一项针对真实高度安全的窄对照候选，不靠改gate宣称有效或启动Cm训练。
现有净力contact-proxy改善不等于可靠抓持；训练所得Cm-on/off utility仍OPEN。
GPU6累计691.45s（含初始化），均已结束，36tests通过。
[per-finger实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-per-finger-control.md)。


## Task ref8：GT consequence 对任务信息的保留

复用ref7的854随机窗口，不再采集；step8 E12/I14预测step9..32任务，
688train/166test、125/31环境隔离，六主模型及三预先声明signed-force扩展同容量/init。
GT预后PROMISING：HEI主误差0.73135→0.39311（46.25%改善，环境bootstrap95%
34.02–57.03%），物理高度失败误差改善32.52%（11.24–49.92%）；H+a也有18.99%
主误差改善。跨半包、留一臂PCA3/ridge1桥的EI接触/高度失败误差相对zero改善
21.65/16.14%，相关0.550/0.723；是共享zero的有噪声边际估计，不是因果中介识别。

完整充分性UNCLEAR：HEI再加a主/物理误差点gain−0.33/−0.93%，但one-sided95%
上界7.96/9.64%仍超过预设5%，不能排除有意义的剩余action信息。Signed扩展下a
又有8.68/12.91%增益，末50epochs loss仍降约40%，初始q/dq经PCA与泛化误差均限制判断。
无泄漏/关键工程bug；root和独立审查复算通过，逐指PD/q/真实tip幅度表附新卡。
下一项有决策价值的是独立冻结的conditional predictability（H+a→E/I，再与直接Ha
比较预测consequence的task增益），不以GT预后或点估计等同链闭合，不立即selector/PPO。
本轮未训练Cm；训练所得matched Cm-on/off utility仍OPEN。
GPU6固定拟合5.56s结束，首CUDA初始化前失败0.76s留存；新产物<4MiB、41tests通过。
[GT consequence实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-gt-consequence-sufficiency.md)。


## Task ref9：conditional consequence prediction 与 OOF 任务价值

冻结ref7数据及ref8的688/166环境隔离划分，E12/I14、intended one-hot保持；
21个固定1500epoch拟合，严格fold-only PCA/尺度/OOF，保存directHa和两个shuffle对照。
总体UNCLEAR，预设A/B/C均未过。Ha I误差比H改善8.50%但区间跨零，E反而恶化20.79%；
冻结Ha置换test动作使I误差增加11.96%(95%4.17–22.24%)，保留局部动作敏感性。
同H候选I差分corr.293、方向62.29%、幅度比.369，未稳定保留动作差异。

OOF后果P_Ha主MSE.6730，弱于directHa.6478；Ha+P_Ha .5995有7.45%改善点估计但
CI跨零。GT同预算oracle .4080仍强；预测后果保留31.18%oracle gain（CI2.90–56.56%），
不能等同超过directHa。跨half预测已做，83行子集GT contrast rank13/9记为NULL/UNCLEAR，
未用伪逆制造逐臂结论。全部14臂预测／GT向量及逐指PD/q/true-tip幅度表在新卡。

按用户要求独立review反常结果，root GPU全权重重放精确一致，无时域/OOF/尺度/shuffle bug。
预测器明显劣于train-mean和persistence，train接近零而test高；末期loss下降不支持追加epoch。
Shuffle较高原sign受共同zero偏移影响，中心化后corr−.076；不救援主结果。
结束当前固定拟合，不selector/PPO；下一Decision应区分受控泛化训练与缺少信息，非更多
memorization。没有关闭Cm核心假设，训练所得matched policy utility仍OPEN。
GPU6主运行59.99s，smoke3.93s，审计仅inference，均结束；新增<12MiB、46tests通过。
[conditional consequence实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-conditional-consequence.md)。

## Task ref10：几何动作表示与物理残差

按ref10将决策native q/PD指令经explicitroot×FK转成120surface对应点流，
共同提供baseline nominal geometry，对比State/Arm/Joint/Flow和shuffle；
fold-only PCA/尺度、严格environment OOF、固定α32ridge，无新仿真。
原PCA/bilinear残差screen UNPROMISING：Flow I12.440明显崩溃，testshuffle反而
降低误差。按用户要求独立review：无来源/标签/泄漏/单位/求解bug，root GPU全重放0误差。
单窗口/前五窗口贡献57.84%/85.13% Ierror，state×flow乘积超出训练支持并放大输出。

另预先冻结六fit尺度×乘法因子诊断（明确post-result、非独立确认）：移除products
I1.254，固定20mm additive1.138；仍弱于State1.074/Arm1.055/Joint1.041/mean.891。
当前PCA/ridge名义端点路线UNPROMISING，不能否定物理几何动作或OI-CmV2空间网络。
独立NumPy复算六fits/bootstrap一致。GT仍改善task31.50%(CI8.32–51.03)，
P_Flow .7149却输H .6540/P_State .6132，R−.2955；C=true仅胜崩溃directFlow，
root明确不认作uniqueCm收益。不selector/PPO、追加epoch或删outliers。

保留FK/continuous surface合同、OOF工具、14signedarm向量、逐指PD/实测q/true-tip
及名义surface幅度表；后续若继续几何路线，应检验localcontact/sharedspatial归纳
偏置与nominal→realized执行合同。全局matched trained-policy Cm utility仍OPEN。
GPU6两主run最终manifest2.88s/1.19s，首启动hash路径错误在模型计算前终止并留存，
审计只inference；新增<85MiB，50Task tests通过，无剩余本轮GPU进程。
[ref10主实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-geometric-innovation.md)，
[归因实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-geometric-support.md)。

## Task ref10 续接：局部 OI-CmV2 空间后果

续读原会话后补上尚未测试的空间路线：复用V13局部交互/token/fusion模块，
固定width32/4tokens/2cmKNN8，名义PD端点点流与persistence残差；全部854窗口，
原688/166环境划分，严格三折OOF，20后果+8同预算直接/GT/预测后果task模型。

本固定合同UNPROMISING：Flow I.8891，State.8837、Joint.8670、mean.8907；
动作shuffle仅增加I误差.565%，I contrast corr.239/sign53.14%/幅度比.066。
P_Flow主误差.5442与directFlow.5468接近且输P_State.5329，五gate均未过；
GT仍改善主误差30.22%(CI11.98–43.15%)。误差稳定，无旧bilinear爆炸。
全部28保存权重、OOF和15Flow候选重放精确一致；独立review与root核验通过。
99.06%窗口有空间边，动作改变邻居/特征且梯度非零；不是动作未接入。

Root停止这个固定fit，下一Decision需区分名义端点与可预测实际执行，不能靠更多
epoch/seed重复弱动作通路。端点/实际tip运动已有明显描述性差距，但不能唯一归因
反馈抵消或断言空间模型不可学。没有selector/PPO或正式科学结论；全局Cm策略
utility仍OPEN。GPU6主162.62s、smoke6.17s、重放3.44s、review GPU.74s；新增约154MiB、53tests通过。
[空间后果实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-spatial-consequence.md)。


## Task ref10 续接：真实／可预测执行几何

固定8个ridge执行拟合、2个实测oracle与3个可预测空间后果臂；854窗口、原环境
划分、执行训练输入严格source OOF，无新仿真。原全部18q裁剪发现实现错误：URDF
腕部3角为continuous，±pi仅PD尺度fallback。root与独立review确认，保留原run，
修复后重放8ridge、复用2oracle/4control，只重跑3个受影响300update空间臂。
错误投影的预测结果不作为原定连续执行路线负证据。

修正后Ha finger误差比H降低79.81%(CI75.70–83.48)，q18 .4793低于nominal .6090；
但surface XYZ-component RMSE34.09mm vsnominal32.30mm。无拟合FK分解发现真实
腕部+预测手指仅2.13mm，预测腕部+真实手指33.94mm，说明事实端点误差主要来自
腕部；前5窗口贡献83.45%误差，未删样本。名义／当前腕部的因果替代分别32.15／
50.92mm，没有取得oracle精度。不能据此宣称执行不可预测，也不能用oracle部署。

本固定端点空间合同UNPROMISING：PredSurface I.8972差于State.8837/PredJoint.8620，
I contrast corr.152/sign50.86%/幅度.0235、testshuffle penalty.301%，五gate均未过。
实测Surface I.8913未胜nominal；实测Joint.8415比State改善4.77%(CI1.17–8.06)，
仅post-treatment诊断，支持研究表示传递而非断言信息不存在。全部保存权重/候选
重放0、ridge残差≤6.66e−16，独立review重建metrics/bootstrap/OOF一致；动作通路有效
（99.30%窗口有边、test22.81active points），57tests通过，原+smoke+repair约154MiB。
Root停止反复拟合totalendpoint合同，保留可预测finger动作信号；后续区分共同腕部
演化和finger相对几何，或检视spatial瓶颈。没有selector/PPO；全局Cm utility仍OPEN。
[执行几何实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-execution-geometry.md)。

## Task ref10 续接：动作传递与按手指残差

冻结空间各阶段做24个source-only线性动作decoder：RawHandBase恢复已知输入
nominal≈1.000/forecast.9987，trainedFused仅.3306/.3436；当前有接触支持的
手指子集也只有.4122/.4275。该诊断只说明固定14arm线性恢复弱，不证明物理
泛化或2cm半径丢失：arm均值lookup自己已≈1/.996，LocalFlow均值代理省略
实际边的位置、法向、距离与个别flow。独立复算通过，root收紧结论范围。

据此执行七头匹配factorial：完整按手指mean/RMS相对点流／相同forecast关节，
普通／15候选均值中心化残差；共同H156，复用三环境OOF State，固定300update。
本固定合同UNPROMISING：PredFingerCentered I.9124差于State.8837（gain−3.24%，
CI−5.94..−.62%），也未优于匹配PredJointCentered.9077。测试动作打乱罚2.75%
但CI跨零，I contrast corr.1565/sign56.57%/zeroMSEgain1.58%，三gate均失败。
名义几何中心化相对plain改善3.91%但CI跨零；不把更大响应幅度当作方向正确。

完整H/action/source-norm、四State和七新head/候选重放0；中心化均值约1e−7，
独立review复算metrics/contrast0、bootstrap≤2.39e−7，未见影响负结果的实现错误；
source-OOF/full-test nuisance差及中心化不能修正共同偏差保留为限制，未唯一归因。
62Task tests通过。GPU6 fidelity主11.67s、factorial主9.30s，零新仿真或策略训练。
Root停止这份mean/RMS+中心化fit，不追加epoch/seed；下轮须区分conditional
response、state nuisance或详细contact信息，不能据此关闭全部空间Cm。
两项都不提供strict nested consequence OOF或matched trained-policy utility；
North-star Cm utility仍OPEN。
[动作传递诊断](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-spatial-action-fidelity.md)，
[按手指动作残差](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-relative-finger-innovation.md)。

## 暂停旧执行路线，在原分支推进 ref11

最后一个surface-relative/current-wrist物理basis × Physics/OOFState nuisance
matched ridge32 Probe为UNPROMISING：PhysicsContact I.98857与同配置State.98882
接近，输priorState.88370；OOFStateContact1.03884。动作shuffle罚2.512%
(CI+.279..+4.959%)，raw Icontrast corr.4047/sign62.86%，四gate未过。
Root重放0、独立saved-matrix/statistics复算通过，未见影响负结果的实现错误；
动作scale floor导致不同有效shrinkage，current-wrist/采样normal/有限basis
限制保留，不否定所有contact geometry。GPU6主3.34s，零新仿真或policy训练。
[收尾实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-contact-innovation.md)。

用户澄清“收尾”仅指暂停旧执行预测路线，继续在原分支
`agent/cm-interaction-oracle` 按Task ref11推进，不另建分支：先以真实hand point-flow为
oracle action，分离 `(H,F_hand)→E/I` 规划问题与以后 `joint→desired flow` 控制。
已完成State、GT endpoint、GT0→4/4→8 chunks及matched shuffled flow比较，
先只看E/I；不继续execution、候选nativearm或任务Y拟合。实际flow是
post-treatment信息，正向只能支持oracle表示Probe，不自动证明前瞻planning或
同状态候选因果contrast。最终matched Cm-on/off训练策略utility仍OPEN。


Ref11首个raw720oracle-flow Probe已完成（原分支，代码452d7e5）。同688/166
windows和125/31environment split，五臂同59066参数/300updates。Chunk E.48968
vsState.52895，改善7.42%（95%CI+.46..+14.04%），E门槛通过；I.55603
vsState.60486，改善8.07%但CI−1.25..+17.62%，预设联合E/I gate为UNPROMISING。
保留E的PROMISING探索信号与I不确定性，不据此关闭oracle flow路线。
trained/frozen shuffle显示action敏感性；Chunk-vsEndpoint CI均跨零，时序优越性
未建立。新H包含清晰current q/dq/geometry，旧State.8837与新State.6049不属
同合同，跨run提升不能归因于flow。test0同状态配对，candidate contrast为UNCLEAR。
完整GPU input/FK/weight/statistics重放误差0，69Task tests通过；GPU6主5.62s，
零新仿真或policy训练。只读检查确认已有restore仅支持coldstate，不能恢复warm
PhysX cache。用户明确暂不做配对数据及相关工程检查/采集，列入延后证据。
独立CPU权重/源统计/bootstrap复算通过（归一化误差≤1.34e-6），报告归档；
当前单seed预测Probe完成。不放松门槛、不追加seed/epoch，也不恢复旧execution
预测支线；尚未完成多seed泛化Validation或真正flow-space planning。
[Oracle flow实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-oracle-hand-flow.md)。


## Ref12：oracle flow → OOF E/I → Y 链路

按用户ref12在原分支完成现有数据的matched任务预测Probe，配对数据/相关工程
和execution predictor继续暂停。3环境fold的H/PCA/标签stats仅使用各fit环境；
688source每行仅用hold预测，166test用冻结ref11 full-source模型。六新Cm fits
300updates、六同容量Y heads500updates，E12/I14和ref8step9起Y8保持原义。

直接actualflow→Y误差.45788 vsH.63951，改善28.40%（CI12.23..41.48%）；
预测E/I→Y.54712，改善14.45%（CI−1.57..29.47%）；GT E/I.45440，改善
28.95%（CI10.61..42.74%）。R.4991（CI−.0639..1.2790），固定链路gate
UNPROMISING，保留直接flow与GT的PROMISING任务信息，不否定所有flow/Cm。
Hybrid.43694相对直接flow改善4.57%但CI跨零，unique Cm增益UNCLEAR。
OOF source Chunk I.61550 vsfull-source in-sample.39092，后者从未进入Y训练；
OOF/full-test分布与有限优化预算保留为限制，不唯一归因于任何机制。
GPU6主11.72s；全部input/foldnormalizer/FK/权重/OOF/统计重放误差0，零新仿真。
独立CPU核对所有8个Y标签、sourcefold统计/权重/OOF/shuffle/bootstrap/R，
归一化replay≤2.17e-6，未见影响结果的实现缺陷；72Task tests、仓库验证通过。
不追加epoch/seed或进入planner；未来改变模型须另立可判别的Decision实验。
全局matched训练策略Cm utility仍OPEN。
[Oracle flow任务链实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-oracle-flow-task.md)。

## ref13：Oracle-Y Utility Gate 已完成

用户 ref13 重新启用同前缀配对候选采集，在原分支完成逆向必要性 Probe。
先修复 GPU pipeline/异步前缀污染：最终用GPU PhysX+CPU tensor pipeline、
GPU6 frozen policy，四个同步fork组。32current-only前缀（29s3/3s7）在候选前
冻结；fresh baseline/repeat字节一致，24候选branch前缀差0，32×7结果完整。
原异步46cohort和失败候选保留为工程记录，不作GateA科学证据。

GateA固定合同UNPROMISING：baseline稳定Z23/32，GT-Y选择24/32，gain3.125pp
(anchor95%CI0..9.375)，未达≥5pp/lower95>0；GT-Z候选上限25/32。
仅两处潜在rescue，其中一处由候选顺序平局选中；另一处全部短Y相同，baseline
step89才越过掉落高度阈值，成功候选到step90也仅高于阈值1.838mm。
因此停止当前短Y/固定七候选selector，不把小正点估计当正式控制收益，也不
扩成全部Y、E/I或Cm无效。改变U系数不能拆开相同完整Y标签的候选。
GateB加噪、GateC表示比较和GateD执行/PPO均未启动；旧execution forecast和
进一步MSE调参继续暂停。未来需先冻结具有有意义长期机会的目标/候选合同。

GPU真实point-flow归档14panel FK/live最大tip误差1.714e−5m，Y/Z重建完全一致；
包含样点、代码/mesh和原始panel哈希。独立raw标签/组装/PD/selector/CI复算
全部一致，仓库验证通过。零新fit/policytraining。native累计1679.61s，
产物约420MiB，预算内；78Task/pairedsim+7coldcontract tests通过，1skip。
共享solver四组mosaic、强s3条件群体、forcepair代理及短Z horizon仍限制外推，
不把anchorCI当group数值独立或在线混合选择已验证。最终matched训练策略
Cm utility仍OPEN。[实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-oracle-y-utility.md)。


## Ref13_1 修正：滚动 GT-Y 可分性

完成纯CPU保存轨迹审计，零新模型或仿真：32×7候选的原Y/Z和tau0标签精确
重放，保持32步窗口、原utility、每8步cadence。124个Z不同的候选pair中50个
初始utility平局；46个在滚动后正确区分，涉及8/32原前缀，提前29–47steps，
记录路径可分性PROMISING。这些pair并非独立实验；8前缀中6个baseline本已
成功，env33已由one-shot救回，不能写成8个新增rescue或rolling策略收益。

env33 middle−/grip+在tau16分开、提前32steps；env36 baseline/thumb−在所有
已保存每8步查询tau0..56仍平局，dense诊断tau57/58才差1、提前32/31steps。
原90步记录的tau56窗口终点88，下一查询tau64需未来到96（缺6steps），不能
补齐或把dense诊断当主cadence成功。后续候选H已不同，无法拼接反事实切换。

因此ref13 one-shot no-go保留，明确不关闭rolling short-Y；后续优先新的共享
当前状态rolling GT-Y干预证据，而非直接改long-Y/扩大authority或继续predictor。
实际rolling收益/可执行上限仍UNCLEAR，原GT-Z25/32只是旧分支机会ceil。
主审计1.43s、产物不足1MiB、14合同回归测试通过；源码/数据哈希和边界冻结。
独立CPU全部rolling标签/risk/U、事件/提前量/统计复算一致；4pair缺后续查询，
不额外声称持续性。去掉这4pair后的42pair仍覆盖全部8信号前缀。
旧execution forecast/PPO和MSE调参继续暂停，最终matched训练策略Cm utility
仍OPEN。[滚动审计卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-rolling-gt-y.md)。

## Ref14 主链 / ref14_1 rolling oracle 已完成

用户更新ref14_1，当前只推进真正same-current-state RollingGT-Y Oracle Control。
保持32步Y/U、七个K8候选、每8步重规划与原稳定Z90；新的真实组合执行产生
下一轮状态，禁止候选世界状态拼接。每次fresh冷重放整条实际动作前缀，
PhysX solver历史由重放重建；旧25/32仅历史one-shot机会，不是rolling上限。
原32anchor/四组、已有暴露cohort复用，属于机制Probe，不提供独立Validation。

工程smoke163.16s完成：7anchor新baseline/repeat/旧状态历史动作90步结果
EXACT；4个非零组合动作之后，两次下一状态重放仍EXACT。代码与协议已本地
提交，主运行commit337d3a6，GPU6/7最多4进程（每卡2），<=7200s/4GiB。
固定utility<=1.25，所有baseline达到上界时，baseline优先平局可精确省去
六个候选，不改变选择，不编造其Y；其他情况完整同状态七候选。
主实验外部中断后已由本会话恢复（恢复入口a949b62），原progress保留，当前
进度见同run目录resume_progress.json；原短Y/候选/执行源码与协议哈希未变。
固定32-anchor gate 已完成：baseline 23/32，实际 same-current-state rolling
GT-Y 27/32，救回4、伤害0，净增益12.5pp；paired bootstrap 95%区间
3.125--25.0pp，lower95为3.125pp，达到卡片预先固定的 PROMISING gate。
四组全部完成，baseline repeat/prefix 和 actual mixed path 检查通过；这是暴露
cohort上的机制 Probe，不是独立 Validation，也不能替代最终 trained-policy
Cm-on/off。结果文件为
`outputs/cm-interaction-oracle/rolling-oracle-control-s263-s264/result.json`。
预测器、PPO尚未启动；离线 Y ranking/noise tolerance Probe 已完成：43个完整计划、
339个同状态面板；sigma .10 的 median pairwise accuracy=0.758，sigma .20=0.696，
离线筛选状态 PROMISING。它只提供预测器的排序门槛，不提供 noisy rolling Z
retention；下一步仍需冻结 predictor 输入/输出并做真实 rolling intervention。
原7200s/4GiB主实验预算不重置。
用户授权ref14_2并行工作已完成并本地提交912aa1a：只读完成组数据资产、固定Y
时序诊断、已有flow与tinyFK审计、ranking/noise评估合同；17新测试通过，无
新仿真或训练。[并行交付](../src/task/cm-interaction-oracle/docs/ref/ref14_2_progress.md)。
[固定协议与进度入口](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-rolling-oracle-control.md)。


## Ref14_3：actual-flow learned-Y ranking 已完成

按用户ref14_3进入预测器阶段：已有完整32anchor/fourgroup输出重新打包，
GPU FK重建120点两段actualflow，无新PhysX。四折anchor隔离，source内三折
E/I OOF，matched H/Direct/Bottleneck/Hybrid/GT_EI；保持原Y8和U。
339complete panels/2276strict pairs：Direct57.38%、Bottleneck57.82%、
Hybrid57.95%，均未过冻结70%离线gate，固定合同UNPROMISING。GT_EI67.36%。
三臂flowshuffle退化CI均正，保留动作敏感信号；Bottleneck/Hybrid相对Direct
仅+.44/+.57pp，CI跨零，unique E/I contribution UNCLEAR。
GPU0训练30.79s、savedweight重放3.95s且误差0；独立只读CPU/NumPy核查一致。
H约50%的微小偏差来自跨batch的1.19e−7舍入，仅5strictpairs，原始证据保留。
110Task tests/1skip与仓库changed验证通过。无新增rolling learned-Y、execution
或PPO；不将当前模型、预算、selectedoracle support的负结果升级为全局否定。
Source ranking也仅约.585–.711，下一步最小Decision可另行冻结source-panel
ranking objective比较以区分拟合/泛化问题；不直接扩epoch/seed或消耗仿真。
[Ref14_3协议与结果](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-actual-flow-y-ranking.md)。
最终trainedpolicy matched Cm-on/off utility仍OPEN。


## Ref14 baseline source_e260：冻结 critic 排序已完成

用户指定先比较critic与Y，使用同一source_e260 PPO checkpoint，无新模型训练。
原ref13的32anchor/七候选/fourgroup面板完整冷重放，28候选执行与原H、
动作、physical32以及整个height/pair/valid90 EXACT；独立critic replay误差0。
原PPO奖励含2approach/10held_lift/5lift_progress，已按源公式从保存状态离线
重建；最初遗漏shaping的评分保留为诊断，不作为原PPO评分证据。
同one-shot面板baseline23/32、GT-Y24/32；reward8+gamma^8V8为20/32，
救回1、伤害4；V-only19/32。原冻结gate UNPROMISING，暂不推进这个critic
评分的rolling执行或Cm→V，不改系数寻找正结果。独立组件/排名/统计复算通过，
112Task tests/1skip。重放RNG错误的旧运行保留并排除；修复后总主采集690.254s，
低于原1200s cap。这里只是one-shot潜在结果组合，不能与实际rolling GT-Y27/32
当作同协议对照，也不否定所有critic或全局Cm假设。
[协议、修复和完整结果](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-frozen-critic-ranking.md)。
最终trained-policy matched Cm-on/off utility仍OPEN。


## Ref15：training-only GT auxiliary 有界 Probe 已完成

用户授权以source_e260 PPO为起点，让GT真实h8交互loss直接塑造actor z，
推理仍仅actor；四臂均追加64epoch/131072交互/3072更新。完整96新frame0
episode持续45tick且以后不drop：PPO62、条件GT21、shuffle51、stopgrad62；
source本身3。条件GT较PPO差42.71pp，更多后续掉落。该cohort不同于旧32anchor
GT-Y/critic，不能跨协议比较。A/D整native权重/RMS与逐episode结果EXACT，
独立GT/action/mask/reward/MC/成功统计审计通过，未见实现缺陷导致负结果。

冻结gate为UNCLEAR：条件GT heldout loss仅改善2.24%，未达到5%学习门槛；
同z return readout较PPO改善2.52%，弱于shuffle。原生critic独立于actor z；
readout针对finite source-policy MC与MC-sourceV residual，不是当前策略advantage。
本轮无策略收益，但不能否定已充分学会GT监督的路线。停止固定h8/lambda.05
短配方，不扩epoch/seed/系数、不拟合Cm、不进入Validation。若重开先另冻最小
Decision核实aux可学性/独立覆盖；不把训练reward或loss升级为utility。
训练429.28s/1701.87GPU-worker秒，评价222.44s/552.89GPU-worker秒；117Task
测试通过/1skip及changed验证PASS，原始失败smoke和完整证据保留。
[协议与完整结果](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-gt-interaction-aux.md)。
最终trained-policy matched Cm-on/off utility仍OPEN。


## 新路线：cm-pointflow-effect-pretrain 数据预检

用户明确指定新分支`cm-pointflow-effect-pretrain`，先按Task ref1检查SPIDER
retarget_full是否可用于点流/effect预训练。四来源各一条Inspire，120文件checksum
通过；750帧qpos/qvel/ctrl/time及portable MuJoCo模型加载/FK点流工程检查PASS。
完整Inspire清单1946条，当前只下载4条及全部依赖。尚未开始大规模训练或PPO。

下载SSL EOF已用curl解决。Python3.12/MuJoCo3.7隔离环境可加载原始场景；
原Isaac环境未改。SPIDER18维ctrl不等于现有PPO action，实际轨迹违反当前native
耦合最大约1.017rad，因此native control replay仍需适配/漂移检查，不能假定直接
可用。数据支持样本级离线几何点流提取，不代表Isaac dynamics或策略utility。
[预检及后续数据合同](../src/task/cm-pointflow-effect-pretrain/docs/DATASET_PREFLIGHT.md)。


## 2026-10-06 主工作树删除后的恢复边界

用户确认删除了Ref2Dex-agent-baseline，当前工作树原.git和outputs均指向该目录。
从远程Ref2Dex-agent恢复至1afe075的历史，创建当前目录独立.git并保留
cm-pointflow-effect-pretrain分支；使用index-only read-tree，未checkout或覆盖工作文件。
当前Task未push的提交对象身份未找回，但代码/文档文件还在，保存为恢复快照。
旧.git指针和outputs链接在artifacts/git-recovery-20261006保留，新outputs为本地空目录。
被删除的checkpoint、rollout和结果数组未恢复；此前实验卡所指原始输出现不可访问，
不能把已保留的文档摘要当作原始证据仍可复算。Git恢复不是数据恢复。
本轮OakInk2产物使用Task声明的artifacts/cm-pointflow-effect-pretrain目录。


## 2026-10-06 OakInk2 annotation-only 100-sequence audit

按用户ref2，本Task当前只推进OakInk2，不混入SPIDER/其他数据。100条固定序列及
五个资产包已校验下载，6.29GB；GPU0完成双手11点/物体512点/SE(3)审计877.62s。
646565手帧、477序列物体对，无缺失mesh、无效rigid pose或帧号缺口。
4961958重叠窗口原始静止90.40%；program筛选731579窗口，运动384679，
静止47.42%，支持继续小规模预测Probe。120Hz的h8只有66.67ms，不能沿用30Hz解释。
发现少量相邻物体跳变，影响1344个program窗口；训练前检查并屏蔽跨异常窗口，
按sequence划分、运动/静止平衡，再决定大规模投入。尚未开始预训练/PPO；
数据工程可用不等于可学性、因果effect或策略utility。Git已独立恢复，但历史产物
删除仍影响6个旧测试；新OakInk2流程不依赖这些路径。
[结果与下一步](../src/task/cm-pointflow-effect-pretrain/docs/OAKINK2_DATA_RESULTS.md)。


## 2026-10-07 WM30/K24 架构实现与接口完成

用户指定架构并确认program锚点+0.5m局部物体、三组独立训练及B验证shuffle、
全627条标注/最多4GPU/整组24h。完成1cm稀疏卷积64/128/256、384维8层场景/
4层动作/6层dynamics Transformer、多物体24步SE(3)及解析点损失，37208777参数。
局部选择已核实使用当前mesh几何中心，避免部分标注原点偏离>20cm导致选错。
6项合同测试及12条现有数据三组6步GPU smoke通过，初始化hash一致，checkpoint
保存/恢复/独立推理接口通过。属于工程检查，不作模型质量或policy utility结论。
全量下载（HTTP range续传）与GPU3预处理运行中；627完整校验及split审计后自动
启动GPU0/1/2三组，每组40000更新、有效batch16，共享24h截止时间。不启动PPO。
[实现与运行入口](../src/task/cm-pointflow-effect-pretrain/docs/WM30_INTERFACE.md)；
[固定实验协议](../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-oakink2-wm30-k24.md)。


## 2026-10-07 WM30 全量训练中间状态

627条全部下载/预处理完成，5060616窗口，501/70/56序列划分。实际运行提交ec4d918，
GPU1/2/3三组训练已运行约5h17m；H33310、H+A27796、shuffle28943/40000，
最慢组估计还需2.3–2.5h，仍在24h预算内。数值/进程正常，但最近运动锚点h24验证
点EPE三组均约28.17mm，与静止基线几乎一致，尚无A收益。不同更新数仅是中间观察。
四个验证窗口的checkpoint核对确认实际近零预测（GT最大90mm、预测最大约0.05mm），
标签非零、A梯度存在，FP32同样近零；不足以确定优化原因或否定方法。保持当前实验
到固定matched更新数，不改活跃配置/不追加预算；原始中间检查记录保留在Task产物。

## 2026-10-07 用户指定 ref3 PointWorld-small 替换

用户指定参考本地PointWorld、按Task ref3换架构并启动训练，随后明确授权停止旧训练。
旧三组均保存checkpoint/正常退出，H40000、H+A33464、shuffle34823，不能形成
matched最终比较；原始产物保留。当前分支不变，Mission/claim不变。

新实现直接使用PointWorld PTv3-small、128维/patch128/1cm，统一场景与24步双手
点输入、object pooling+SE(3)小head，逐时域train-only归一化和官方逐帧运动soft权重。
实测50,495,881参数，完整监督保持每物体512点。重复栅格的SparseConv不稳定问题
在正式运行前修复：唯一voxel聚合+inverse恢复，评估全部层固定序列化顺序。
7合同测试通过/1旧模型可选GPU测试skip；三组6步smoke初始化相同且checkpoint精确
保存恢复，独立val推理/恢复加载通过。80步重复运动batch loss3.016→0.225、
末端anchor EPE44.3→16.4mm，只证明学习链路/小batch拟合，不作泛化结论。

复用627条完整数据及原sequence split，GPU0/1/2三组独立40000更新、有效batch16、
共享24h上限。运行目录outputs/cm-pointflow-effect-pretrain/pointworld-small-wm24-20261007，
启动前统计已冻结、接口已验收；实时进程/进度/运行提交见group_status.json。
实际运行提交f507f18，launcher3942111，worker3942116/3942117/3942118；三组均已
完成正式更新，loss/梯度有限、初始化/data/stats/config身份一致。早期每步约1.2–1.4s，
40000步粗估14–16h，共享24h上限不变。本次改动范围verify通过；全恢复分支仍有
6个依赖已删除baseline产物的历史测试失败（另102项通过），不影响新路线输入。
不启动PPO；最终trained-policy matched Cm-on/off utility仍OPEN。
[协议及证据](../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-pointworld-small-wm24.md)。

ref4只读诊断已完成：229训练序列/512窗口，528action点平均合为160.55个空间voxel，
同手同关键点跨时间碰撞影响83.70%的点，下一版优先保留时间身份。5mm逐帧selector
原始权重偏低，但归一化后moving-anchor相对均匀监督系数中位1.217，不能据此声称
所有moving监督仅剩2%–3%；部分旋转样本仍被相对降权。审计自身遗漏padding mask
的初次归一化份额已排除并修正，原训练mask正确；三组继续原配方/原预算，结论仍UNCLEAR。
[诊断协议、结果和修正边界](../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-pointworld-ref4-input-loss-audit.md)。

## 2026-10-08 consequence twin contract hardening

在独立工程复核后，twin v1 继续收紧：native prefix 按 30 Hz control tick 与 60 Hz
physics frame 分离计数；zero-step replay 只接受自然的空列表，显式空二维
action 仍核对 18 维；RNN 必须使用显式 `is_rnn/state` sentinel；Python/NumPy/Torch RNG
格式、非空 physics/history/controller provenance 均在入口拒绝缺失值；native adapter
自动冻结 direct tensor、scalar 和 Enum（包括 `_state_init`）inventory，并提供 CPU/GPU
Torch RNG restore helper。Task tests 为 114 passed，compileall 与 diff-check 通过。当前仍没有 native branch runner
和真实 twin branch 数据，不能把合同测试当作 twin coverage 或 evaluator 科学证据。
