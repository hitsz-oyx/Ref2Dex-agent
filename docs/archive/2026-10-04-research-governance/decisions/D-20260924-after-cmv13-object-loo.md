# Decision Memo — V1.3 object-LOO 门失败后（2026-09-24）

Decision: 是否回到单对象强 baseline，重新设计任务对齐、动作条件的 Cm，
还是停止 Cm policy-utility 主张？当前 V1.3 路线已按预注册门停止。

Key evidence: 五对象均有足够 H10 正负状态，但冻结 V1.3 高密度 token
的 object-LOO AUC 仅 0.6387；action-blind 为 0.6694，action-shuffled
为 0.6382，10135 点相对 1538 点仅 +0.0025。此前局部 reward、ranking、
auxiliary 和简单 x/z 序列也未产生稳定 policy gain。问题不是再加点流，
而是当前表示/监督目标没有保留对抓取决策有用的动作信息。

Option A — 回到已强的单对象 self-trained baseline，用随机执行数据先验证
“手指闭合保持接触 + 腕部上抬”动作族对 H10 承重成功的真实效应；仅物理门
通过后，训练显式 action-conditioned、以承重结果为目标的新 Cm，并做小规模
matched on/off/placebo。预计设计/物理 Probe 1–3h、1 GPU <=30min；正向后
matched Probe 需 <=2 GPU、数小时。成功：保留核心 Cm utility claim；失败：
有依据停止任务对齐 Cm 路线并复盘 claim。

Option B — 停止新增计算，修改或放弃“Cm 改善策略”的核心 claim。计算成本 0；
这会改变 `MISSION.md` 的最终目标，需要用户明确决定。

Option C — 用当前五对象 RL 状态重训 V1.3。预计至少数小时到一天并需构建新
监督缓存；当前 action-aware 不优于 blind 的证据使其成功概率低，不建议。

AI recommendation: A。跨物体泛化是 P3，不应继续阻塞 P1/P2；先在最强、
最可控的单对象任务上证明动作条件 Cm 能否带来 matched policy utility，信息
价值最高。若新的受控动作族连真实物理门都不过，再选择 B，而不是继续换头。
