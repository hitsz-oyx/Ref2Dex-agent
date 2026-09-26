# Ref2Dex Current Research State

Updated: 2026-09-26

本文件是新 agent 的默认入口。运行细节、seed、分数和失败路径只保留在
对应 experiment card；搜索预算和 family 状态在
[`RESEARCH_QUEUE.yaml`](RESEARCH_QUEUE.yaml) 中维护。

## Current decision

用户于 2026-09-26 曾授权新的 HF05 goal；该 CPU-only selective causal gate
已完成并判定 `UNPROMISING`。它只在 1/126 个 holdout 状态介入，held-lift
没有超过 always-base，coverage/policy gate 失败。HF01–HF05 均已冻结，不启动
新的 GPU、PPO、online 或 collector。结果见
[`P-20260926-selective-causal-gate.md`](experiments/probes/P-20260926-selective-causal-gate.md)，
路线处置见 [`D-20260926-after-hf05-selective-gate.md`](decisions/D-20260926-after-hf05-selective-gate.md)。

## North-star scoreboard

| 目标 | 当前状态 | 证据边界 |
| --- | --- | --- |
| Self-trained grasp | `PARTIAL` | 固定物体身份/观测路由已有探索性非零抓取，但还不是单一观测驱动 actor 的稳定结果。 |
| Cm one-step information | `PARTIAL` | 随机动作干预中有可学物理效应；信息依赖表示、分布和目标。 |
| Cm policy utility | `OPEN` | 尚无跨训练 seed 的 matched Cm-on > Cm-off 证据；effect-rank 正式 Validation 的正向主张已 `REFUTED`。 |
| Generalization | `OPEN` | 未见物体和多轨迹上的 Cm 收益尚未建立。 |

最终研究价值由第三项决定：在足够可用的 self-trained substrate 上证明 Cm
对真实策略决策有因果增益，而不是只提高离线预测指标。

当前阶段聚焦固定自训练任务分布内的 Cm policy utility；跨物体泛化暂不作为
本阶段门槛。见 [目标重述](decisions/D-20260925-cm-goal-reframe.md)。

## Confirmed long-term facts

- 当前可靠的抓取 substrate 是自训练专家/层级路由；它仍可能读取特权物体
  身份，不能报告成完整 GRAB actor。
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
| `HF05` selective-causal-intervention | `C3` | `UNPROMISING` | 1/1，CPU gate failed | `agent/cm-selective-causal-gate` |

新 Probe 必须登记一个 family、递增 `probe_index_in_family`，并通过
[`RESEARCH_QUEUE.yaml`](RESEARCH_QUEUE.yaml) 的预算门。family 用完预算仍无
信息增益时，必须切换高层假设；换 metric、horizon 或 seed 不会重置预算。

## Next step

HF01–HF04 的实验卡、manifest、结果索引与 Git 提交已完成一轮只读复核，见
[closeout audit](handoffs/HF01_HF04_CLOSEOUT_AUDIT_20260926.md)。HF05 的唯一
existing-record screen 已完成并失败固定 policy/safety gate。当前不安排新的
Cm 实验；不得恢复 HF01–HF05，也不得换阈值、seed、target 或启动 online/PPO。

主代理复核发现 baseline 注册 thread 在冻结决定之后再次发起 GPU 评估；
相关提交暂不合入 `main`。见
[监督审计](handoffs/BASELINE_POSTFREEZE_PROBE_AUDIT_20260926.md)。
