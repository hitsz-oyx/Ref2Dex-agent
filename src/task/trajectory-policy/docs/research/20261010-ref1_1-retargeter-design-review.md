# ref1_1：真实动作监督的 retargeter 值得先验证

日期：2026-10-10。设计输入：[用户 ref1_1](../user/ref/ref1_1.md)。
论文事实单独见 [PointWAM 来源核对](20261010-ref1_1-pointwam-source-check.md)；
本文是针对当前仓库的判断与待检验方案，不是新方法已经有效的结论。

更新：用户询问已有研究后核对发现，旧Task已有actual τ+q/dq→native action的
learned Transformer及v2 context版本。下表“当前实现”仅指最近实际运行的TauTracker，
不能代表全仓库没有动作监督R。新Task是历史失败路线的有界后续，不是首次提出。
具体旧实验/代码/结果与下一步去重见[已有研究核对](20261010-retargeter-prior-work.md)。

## 判断和当前证据

建议把近期最小实验优先级转到实际手轨迹—动作监督的 learned retargeter，暂不
启动第三次288维trajectory PPO。不复制整套PointWAM预训练/语言/场景大模型，
也不根据当前失败关闭trajectory接口或宣称现有R就是主要故障。

当前证据有两个方向：GT/dense τ 经原R在最新校准达到4/4、3/4，说明原R对已知
可执行轨迹有效；纯H启动预测和PPO改进弱，warm/final同噪声只有相同2/28稳定成功，
说明高层尚未学出可用行为。新R可能扩大容错/覆盖，但不能自动修复错误的高层意图。
真实Adam回放还发现约95% KL来自轨迹后16步、99%来自XYZ；这是当前参数化/优化
值得重审的线索，不是后半段无效或retargeter瓶颈已被定位的证据。

| 问题 | 当前实现 | ref1_1 中值得检验的变化 |
| --- | --- | --- |
| τ→A 的监督 | hand-only几何拟合加任务训练的冻结R | 同一rollout实际future hand与实际applied native action配对 |
| 高层输出为何可执行 | 轨迹初始化/真实奖励间接约束 | hand loss加经过learned R的action loss，并审计梯度链 |
| 部署误差 | predicted τ域下失败，GT校准有效 | GT/扰动/实际predicted τ分别测量，不以GT表现代替部署 |
| 状态反馈 | 原R每native control读取实时状态 | 新action chunk必须明确执行长度和重读状态频率 |
| World modeling | 后续计划action-conditioned physical token | PointWAM共享scene辅助监督可借鉴，但不能替代条件WM或Cm因果验证 |

## 数据合同先于换网络

基础样本固定为

`(s_t, actual hand[t+1:t+24]) → applied native action[t:t+7]`。

state包含执行时可获得的q/dq/hand/object pose/velocity，不读取未来物体、参考q、
phase/clock或触觉。τ是同一rollout真正发生的未来手运动，不能混用desired plan、
actor latent或几何拟合q作为“实际未来”。目标使用进入native pre-physics的applied
command，区分它与PD target；本手18维command中12个active坐标、6个passive零列。
保留既有控制单位、耦合、限幅和执行记录，不直接沿用DexJoCo动作合同。

已有两轮trajectory PPO保存了这些字段，先做CPU合同/窗口库存审计，不需要新
物理fork或先采大数据。它们大部分是失败轨迹，适合检查inverse supervision，
不能把回归这些动作当成可用任务示范。旧成功teacher packet明确
`engineering_only=true, training_allowed=false`，不能转成R训练标签；若需要
competent coverage，在本Task重新采有明确训练合同的有界native rollout。

按整个source episode划分split，再生成窗口和train-only normalization。
不得跨reset；保存的before-state序列缺少terminal hand时，排除末端未来不足窗，
不借下个reset帧或padding伪造τ。相邻滑窗不等于独立示范，不将同seed、同初始化
的多env当多seed Validation。无须只留成功：失败paired动作也可用于R，但高层
任务生成仍需competent data/真实奖励和明确goal。

## 三种取舍及建议顺序

1. **继续现288D PPO，仅恢复proposal LR。**有机制正向线索，成本低，但即使
   update位移提高到2.6倍，绝对量仍很小，也没新增示范/执行鲁棒性。保留备选。
2. **先独立学习真实动作retargeter。推荐的下一最小Probe。**隔离τ→A是否更容易
   学、是否依赖τ、是否能执行；保留原R为对照，不同时更改scene输入和高层。
3. **直接重建完整scene+goal联合forecaster/retargeter。**更接近参考的完整路线，
   但同时改变数据、输入、低层、监督，无法便宜定位失败。第二步有物理信号后再做。

第一版R可用小型Transformer：编码24个future hand step及current state，8个action
query预测真实native chunk。网络大小不是PointWAM结果的可迁移保证。训练监督
覆盖actualτ和真实扰动动作产生的τ'；人工τ jitter仍配原A只能称输入误差增强，
不能说它是执行新τ的真实动作标签。轨迹不能唯一确定preload/PD，需检查近似τ的
标签冲突；L1低或network能run不代表抓持成功。

先保留每步读取实时state的闭环习惯：可预测8步但每次仅执行第一步并重读state，
或明确做更短前缀；完整8步开环应另作干预，避免把“换R”和“减少反馈”混成一项。
从GT actualτ开始核验有意义，但GT仅oracle上界，不能部署。

最小后续Probe需三个层次：

- 离线：整episode held-out动作误差，matched-capacity state-only和τshuffle，
  检查R有没有只从state复制行为、忽略τ；所有学习/归一化统计只用train。
- 物理：held-out实际可执行τ再执行、不同可执行τ的控制响应、长时held与后续drop。
  不声称parallel同reset即exact hidden-state fork；重复/控制细节先冻结实验卡。
- 部署：旧forecaster冻结的predictedτ原样进入新R，报告startup、tracking、clip和
  stable success。GT过而predicted未过，应分辨高层误差/覆盖不足，不能归因于全路线。

只有这些Probe有正向物理信号，再联合训练`F(scene,H,G)→τ→R→A`，同时保持手轨迹
语义监督，显式核对action loss确实更新F、没有detach或隐藏动作捷径。单任务G可
固定为“抓住并保持抬升”，无需加入time/phase；常量goal本身不会解决当前优化问题。
scene先用真实当前几何，不默认下载human百万数据或部署语言大模型。

最终仍保持根级Mission：自训练可用策略后，做matched Cm-on/off **训练**收益。
PointWAM的scene辅助head不以任意候选τ为条件；不能把共享scene预测直接称为已
获得`(H,τ)→z_phys`或已完成本项目Cm utility。

## 运行与结论边界

本次只读来源核对和CPU数据合同统计，不运行新训练、采集或仿真。
库存工具：`tools/audit/audit_retargeter_data_contract.py`，输出位于
`outputs/trajectory-policy/retargeter-data-contract-20261010-r1/`。
统计是工程合同证据，不给方法PROMISING标签；后续科学Probe另写Task-local卡片，
明确数据来源/训练资格、split、R反馈频率、错误τ评价、预算和停止条件后执行。

## 本次库存审计结果

`9f74302` CPU合同审计完成：两轮PPO各46,688合法窗口，合计93,376；
每轮80个完整environment episodes加16个部分episode，全部保留真实失败行。
每轮5个完整542步批次可各切518起点，末尾352步partial可切328起点，均16env；
排除跨reset/缺失末端hand的起点，未生成或训练张量数据集。
actual requested/applied command完全一致，passive native command列为零，
q/dq/hand/object/objectvelocity字段齐全且有限。两源seed都是293，旧lambda.95
完整稳定held仅1行、lambda1为0；不能把93,376窗口当独立示范，或把这些失败丰富的
轨迹视为已具备competent数据覆盖。teacher packet资格检查为不可训练。

这是可以立即复用日志格式与一部分inverse配对的工程证据，不是learned R有效。
下一最便宜科学决策是小型R的held-out动作/τ使用性Probe；物理学习前需有清楚
competent coverage和训练资格，必要时用本Task有界native采集补足，不能静默改
旧teacher packet资格。没有启动新R/forecaster/高层PPO/WM训练。
