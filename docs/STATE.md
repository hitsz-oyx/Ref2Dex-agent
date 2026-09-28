# Ref2Dex Current Research State

Updated: 2026-09-28

本文件是新 agent 的默认入口。运行细节、seed、分数和失败路径只保留在
对应 experiment card；搜索预算和 family 状态在
[`RESEARCH_QUEUE.yaml`](RESEARCH_QUEUE.yaml) 中维护。

## Current decision

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
蒸馏。North-star scoreboard 保持不变；当前只推进一个受限的 scratch Cm CPU
support/calibration gate。

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

新 Probe 必须登记一个 family、递增 `probe_index_in_family`，并通过
[`RESEARCH_QUEUE.yaml`](RESEARCH_QUEUE.yaml) 的预算门。family 用完预算仍无
信息增益时，必须切换高层假设；换 metric、horizon 或 seed 不会重置预算。

## Next step

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
teacher label 的六专家覆盖不足，故当前由 `agent_cm` 执行
`CM-SCRATCH-TA-20260928` 的 CPU-only contract/calibration gate；它只能使用现有
transition 字段，禁止猜测缺失的 object-local one-step delta、伪造 episode ID 或
把 `source_e260` 当静态 fallback。若 gate 失败，下一步是按 handoff 中的最小字段
要求重新申请一次受限 support collection；若通过，才进入 matched `cm_on`/`cm_off`
离线 student Probe。
