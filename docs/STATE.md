# Ref2Dex Current Research State

Updated: 2026-09-30

本文件是新 agent 的默认入口。运行细节、seed、分数和失败路径只保留在
对应 experiment card；搜索预算和 family 状态在
[`RESEARCH_QUEUE.yaml`](RESEARCH_QUEUE.yaml) 中维护。

## Current decision

当前执行 HF08 physical-value：用户已批准完整设计，复用单一自训练 source_e260，在 canonical airplane s3/s7/s9 上采集完整转移，比较 plain PPO、direct Q 候选监督和 Cm＋V 候选监督。HF06/HF07及六专家路线的下述记录为历史背景，当前不重新选择路线。

2026-09-30 用户明确下一阶段优先回答 **Cm 策略价值**，复用当前已验证的
六专家底座；单一 actor 蒸馏不作为默认优先交付。交付期限更新为
2026-10-03 23:59（Asia/Shanghai），资源边界仍见 `CAMPAIGN.md`。
本次用户授权暂时跳过代理工作流搭建，聚焦研究推进。交付标准已明确：先取得
真实策略上的 matched Probe，出现正向信号就优先完成正式 Validation；本记录
不代表新的实验结果。

本轮固定 scratch MLP 可行性检查已完成，teacher-envelope 的 contact coverage
和 arbitration stability 未过门，HF06 关闭，不调参。HF07 随后在固定三条 airplane
motion 上完成了四臂真实策略 matched Probe：on 18/128、off 18/128、random
21/128、action 14/128，八个 native arm、配对和初始化合同有效。该 BC 推理输入
实现判为 `UNPROMISING` 并关闭，不升级 Validation。它仍需 self-trained expert
在推理时生成候选动作，只有一个 BC training seed，不能否定整个 Cm 思路。
见 [实验卡](experiments/probes/P-20260930-cm-inference-bottleneck.md)。
North-star scoreboard 不变：Cm policy utility 尚未证明。本轮实验已结束，无本轮
训练/评估进程遗留；下一步返回高层机制选择，不扫描上述局部实现。

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

2026-09-30 已在用户授权范围内并行派发两个受控 Probe：
`T-20260928-cm-calibration-repair-screen-r1` 做一次 fit-only CPU 校准修复检查，
`T-20260930-six-expert-trajectory-distillation` 独立审计并尝试真正的六专家逐步轨迹
蒸馏。前者预算为 2 CPU、15 分钟、1 GiB，后者为 1 GPU、60 分钟、5 GiB；两者都不
产生正式 Cm claim。旧 r2 只是 ridge 加 in-sample residual screen，不能据此否定设计
MLP 或整条 Cm 路线；校准修复若仍失败则停止局部 calibration tuning，蒸馏继续独立
推进。当前结果以各自 Broker handoff 为准。

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
| `HF01` local-effect-ranking | `C3` | `KILLED` | 3/3，冻结 | `agent/cm-option-value` |
| `HF02` temporal-cm | `C3` | `PAUSED`（slot-2 UNPROMISING） | 2/3 | `agent/cm-temporal` |
| `HF03` contact-supported-credit | `C3` | `KILLED`（Probe UNPROMISING） | 1/1，CPU gate failed | `agent/cm-contact-credit` |
| `HF04` trajectory-level-credit | `C3` | `KILLED`（Probe UNPROMISING） | 1/1，CPU gate failed | `agent/cm-trajectory-credit` |
| `HF05` selective-causal-intervention | `C3` | `KILLED`（Probe UNPROMISING） | 1/1，CPU gate failed | `agent/cm-selective-causal-gate` |
| `HF06` scratch-offline-teacher-arbitration | `C3` | `KILLED`（teacher-envelope UNPROMISING） | 3/3，MLP coverage/stability gate failed | `agent/cm-scratch-mlp-policy-probe` |
| `HF07` physical-prediction-inference-bottleneck | `C3` | `KILLED`（固定 BC 接法 UNPROMISING） | 1/1，真实策略 matched gate failed | `agent/cm-scratch-mlp-policy-probe` |
| `HF08` physical-value | `C3` | `ACTIVE`（工程 preflight） | 1/1，r6 科学采集进行中 | `agent/cm-physical-value` |

新 Probe 必须登记一个 family、递增 `probe_index_in_family`，并通过
[`RESEARCH_QUEUE.yaml`](RESEARCH_QUEUE.yaml) 的预算门。family 用完预算仍无
信息增益时，必须切换高层假设；换 metric、horizon 或 seed 不会重置预算。

## Next step

2026-09-30 路线讨论后，用户认可采用动作条件短期物理转移与长期价值结合的方向，
先在 airplane 上检验收益，允许重新设计 Cm，不要求沿用原几何模型。最终交付
要求 Cm 参与策略训练；冻结 actor 的候选动作选择收益仅作机制 Probe。成功标准
将包含明确保持时长与后续掉落检查；跨数据集预训练是可选方向。Mission 已澄清
这些边界，尚未固定网络、reward、数据规模、训练算法或启动新实验。

只读接口核查确认已有 source_e260 的完整首回合轨迹可用于物理模型预训练，但
缺少逐步 reward、显式 next-observation 和 terminal/timeout 区分，不能直接视作
完整 TD 训练数据。当前 actor/critic observation 含未来参考信息，reward 主要是
参考轨迹模仿；新路线仍须明确价值学习目标及与稳定抓取评价的关系。下一步先
完成该设计，再记录 Decision Memo、登记新 family 和固定真实策略 matched Probe。

用户随后委托 root 采用任务相关的长期价值目标：保留参考轨迹引导，加入持续抓取、
保持抬升与掉落反馈，Cm-on/off 使用完全相同的奖励。固定环境交互预算下的持续
抬升成功率为主指标，学习效率为辅，预训练与额外计算成本单独报告。具体奖励
公式、时长、数据预算和算法仍待设计与 preflight，以上仅为已确定研究边界。

具体设计现已写入 [airplane physical-value spec](superpowers/specs/2026-09-30-cm-physical-value-design.md)，
设计已获用户批准，正在实施。它拟用100万条完整过程转移、三臂真实 PPO 与
候选评分的旁路策略监督，以及1.5秒连续保持主指标；成本为两GPU、六小时内，
均是拟定合同而非新证据。只读核查补充确认 source_e260 本身已经有 approach=2、
held-lift=10、progress=5 shaping，不能将源策略表述为只训练模仿reward。
用户随后明确批准具体设计并要求开始工作。HF08 已登记，分支为
agent/cm-physical-value，正在实现完整转移、持续保持评价及候选价值的旁路策略监督。
当前尚未运行新科学实验，后续结果须以原生 manifest 和实验卡为准。

HF08 启动前的历史处置：HF06 和 HF07 均已结束；优先级继续是 Cm 策略价值，期限
2026-10-03 23:59。下一步返回高层机制选择，复用已验证六专家底座；不继续调
teacher envelope 或这次 BC inference-input 的宽度、步数、seed、目标。下面保留
此前路线的证据边界，不将历史派发状态视为当前活跃任务。

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

HF08 工程 preflight 已通过：12项合同/模型/数据检查；真实环境完整采集与评价 smoke；三臂 source e260→e261 的 PPO 更新和保存。三臂初始 actor hash 相同，direct_q 与 cm_value 的行为动作/RNG 完整性及有效 actor 监督梯度标志均为真。以上仅为工程证据；正在检查训练所得 checkpoint 的 actor-only 评价，然后启动公共百万转移采集，尚无 HF08 策略收益结论。当前实施及结果以 [HF08 实验卡](experiments/probes/P-20260930-cm-physical-value.md) 为准。

HF08 保存策略的三臂 actor-only 工程评价均已完成（各6个完整首回合，均从帧0开始）。已启动科学执行 r6，GPU1，固定百万转移公共采集→预训练→六臂PPO→配对评价；总阶段预算6h/20GiB。当前状态以r6原生manifest为准，结论尚未形成。

HF08最新状态：r6首次非空reset接口失败（75.409s），无科学结果。已修复Tensor reset和显式初始reset；r5初始仅覆盖motion0，旧成功评价只作加载证据。r7正在进行重复reset与均衡三轨迹工程检查，科学矩阵待该检查通过；保留原seed与判定门槛，继承失败尝试耗时，不新增Probe slot。

HF08 r7扩展工程检查通过：采集6592条、12个完整episode，三motion各4回合并覆盖重复reset和非零起始帧；actor-only评价三motion各2回合、全部从帧0开始。12项单元合同检查再次通过。r7进入固定科学矩阵，继承r6失败成本，当前为公共采集，尚无科学收益结论。
