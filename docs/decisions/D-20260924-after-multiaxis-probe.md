# Decision Memo — 多轴 Cm 之后的接法（2026-09-24）

Decision: 继续用新 Cm 改造策略训练，还是扩展到受控多步动作序列模型？
两条路线都明显高于当前 Probe 成本，按 `AGENTS.md` 暂停选择。

Evidence: x/y/z 单步真实随机干预在 10 步后产生可重复的物体效应；
新 Cm 在未见 seed163 对 x 效应通过预注册预测门。但固定双轴
在线选择在新 seed164 的 held-lift 为 34/64，低于 base 36/64
和盲 z+ 39/64，已按早停门省略 seed165。Cm 信息存在，不等于
当前单步候选规则能改善完整抓取。

Option A — 把新多轴/十步 Cm 接入 PPO 训练期表示或 critic，
不覆盖在线 actor 动作。预计实现 1–2 h、最小 matched 训练/
评估约 1 h、≤2 GPU。成功：做 Cm-on/off 加置乱目标 Validation；
失败：停止这一训练期接法，不继续调单个系数。

Option B — 收集受控的 2–3 步动作序列效应，训练 sequence-
conditioned Cm 并用于短视域规划。预计 3–6 h、≤2 GPU，
数据量需预先限定。成功：做新 seed 的 matched 规划 Probe，
再进入 Validation；失败：重审局部 Cm 是否足以服务完整抓取。

Option C — 暂停 Cm 扩张，先讨论论文主张与任务设计。计算成本 0；
成功：明确新的可验证 claim；失败：维持当前未证实状态。

AI recommendation: A。B 已提供比旧单轴五步模型更丰富、且
未见 seed 可预测的动作信息；直接在线贪心选择再次失败，
训练期接入能在 actor 的完整时域决策里使用该信息，且成本
低于重建序列模型。但旧五步辅助 Probe 未过升级门，A 仍需
新的预注册 matched Probe，不能预设有效。

User decision: pending.
