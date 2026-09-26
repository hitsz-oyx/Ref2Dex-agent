# Decision Memo — HF02 temporal option Probe 之后（2026-09-26）

## Decision question

HF02 temporal expert-option credit Probe 已在 canonical six-expert、three-airplane route 上完成，但没有通过 policy-value gate。下一阶段是冻结当前 Cm policy-utility search，还是建立一条新的 contact-supported credit 假设？

## Key evidence

1. HF02 fit/holdout collection 合同全部通过：187/186 valid first-episode rows，六臂行数门槛通过，route/checkpoint/motion hashes、start frames、propensity 和 executed-action equality 均通过。
2. Holdout `temporal_cm` held-lift IPW 为 38.710%；相对 `history_only` 为 `+12.903 pp`，相对 `action_shuffled` 为 `-9.677 pp`。Supported-lift 相对两个 comparator 都没有下降，但主要 held-lift gate 失败。
3. `history_plus_expert_id` 的结果高于 `temporal_cm`，说明当前可见信号可能主要来自 expert identity 或 route structure，而不是可迁移的 candidate-action credit。
4. 既有随机干预显示接触阶段存在可学习的局部机制，但接触 expert option 的 lift 增益在 seed234/235 之间没有复现，训练期 H10 auxiliary 路线也没有通过 matched policy gate。

## Option A — 冻结当前 Cm policy-utility campaign（推荐）

保留 self-trained baseline、HF01/HF02 负结果和局部物理机制作为可复核证据，暂不建立新的 Cm policy Probe。后续工作转为论文边界、negative-result closeout 和 Research Debt 整理。

成本是 CPU/文档工作，不消耗 GPU 或新的 Probe slot。若最终研究问题必须证明 Cm 增益，再单独创建新的高层 goal，不复用 HF02 名称或预算。

## Option B — 新建 contact-supported credit 路线

定义一个新的假设：在首次接触后，用 executed-handflow、短序列动作和接触保持/承重标签学习 credit representation，再让它影响训练期 critic/representation 或受限的 contact-stage decision。先做纯 CPU 的标签和 action-alignment audit，只有物理效应在新 seed 上稳定且任务方向一致，才设计新的 GPU Probe。

这需要新的 experiment ID、预算记录和至少一次新的物理 collection；不能通过换 seed、horizon、metric 或模型头继续 HF02。

## Recommendation

当前选择 **Option A**。HF02 已排除“用初次接触时的 temporal expert option 选择直接取得 policy-value 增益”的具体路线；当前不启动 GPU、online、PPO 或新的 collector。

证据索引：

- [HF02 temporal outcome handoff](../handoffs/HF02_TEMPORAL_PROBE_OUTCOME_HANDOFF_20260926.md)
- `docs/experiments/probes/P-20260926-temporal-expert-credit-results.json`
- temporal owner decision source commit `af3d1c0929d125862af11fb33d0c4d72ede22724`
