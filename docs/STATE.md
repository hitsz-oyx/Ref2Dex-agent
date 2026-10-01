# Ref2Dex 当前研究状态

更新：2026-10-01。本摘要整合已交付的主分支与本轮Cm研究事实，不产生正式科研结论，
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
  路线的全局禁止。[队列与路线预算](RESEARCH_QUEUE.yaml)。
- 用户已授权边界内自主选路线，以及六专家蒸馏与新的 Cm 探索；历史标签不因此升级。
- r6 support 的 teacher label 仅覆盖 source_e260，不能证明六专家蒸馏；其缺失轴字段不能回填。
- r7 轴合约通过，但 contact q10 与 delta 覆盖未过 calibration gate，不生成正式 Cm-on 标签。
  [校准证据](handoffs/CM_SCRATCH_CPU_CALIBRATION_R2_AXIS_20260928.md)。

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
HF08 slot不重置。见[修复决策](decisions/D-20261001-native-player-collector-repair.md)。
[HF08实验卡](experiments/probes/P-20260930-cm-physical-value.md)与
[完整结果](experiments/probes/P-20260930-cm-physical-value-results.json)保留边界与数值。
原始数据/checkpoint留在原研究工作树的research/output/P-20260930-cm-physical-value/r7，
未提交Git，不因本次合并移动或删除。
