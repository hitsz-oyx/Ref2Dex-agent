# Ref2Dex 当前研究状态

更新：2026-10-04。本摘要整合已交付的主分支与本轮Cm研究事实，不产生正式科研结论，
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
