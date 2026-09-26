# Decision Memo: effect-rank 现在正式验证，还是先适配 Cm？

Decision: 选择下一阶段主要研究预算的用途。当前所有新结果仍是 Probe，不能升级成论文结论。

为什么现在必须决定：三次短 Probe 已得到正向动作条件信号；下一步无论选择
跨训练 seed Validation，还是改变模型/接法，都需要明显高于单次 Probe 的成本。

Evidence:

1. V1.52 joint Cm-on 对安慰剂 +8.91pp，但对 off 的严格复制门失败；抓取仍不稳定。
2. 新 seed129/130：effect-rank 90/128、joint 83/128、off 72/128、contact-rank 45/128。
3. 新 seed131/132：真实动作 effect-rank 76/128、动作置乱 42/128、off 69/128；
   动作对齐有方向性价值，但 effect-rank 对 off 一正一负。

Option A: 固定现有 effect-rank 配方，开展匹配的跨训练 seed、未见评估 seed
Validation；同时保留 action-shuffled 与 off 对照，预先固定判定门，不调参数。

成本：预计 1–2 小时、最多两张 GPU、新产物约 5–10 GB；先核对当前代码和输入指纹。

成功后：可以开始形成窄范围 Cm policy utility claim，再研究稳定抓取和跨轨迹。

失败后：明确目前 effect-rank 配方不够稳，转入 Option B 的模型/接法重设计。

Option B: 暂不做正式验证，先用当前 PPO 真实转移适配/重设计 Cm 的效应目标
与动作表征，再做小规模 effect-rank 对照 Probe。

成本：预计 1–3 小时探索，且可能需要额外一次 Option A 才能形成正式结论。

成功后：以新方法进入 matched Validation；失败后：返回现有 effect-rank 做验证或换接法。

AI recommendation: **Option A**。已有动作对应关系的正向信号足以值得一次
固定配方的严肃证伪；现在继续改模型会使“Cm 真有用吗”被更多研究自由度遮蔽。
但若 Validation 未过，不将其包装成论文成功，立即转 B。
