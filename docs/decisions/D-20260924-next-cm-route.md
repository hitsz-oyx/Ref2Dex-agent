# Decision Memo — Cm 下一条主路线（2026-09-24）

Decision: 选择训练期 Cm 辅助表示（A），还是扩展多轴/长时域
交互模型后再接控制（B）？两条路线均需要明显超过现有小 Probe
的实现与 matched 训练/评估成本，按 `AGENTS.md` Decision
Checkpoint 暂停，不默认代替用户选择。

Evidence:

1. 自训练 actor 已能稳定提供实验基础；最终缺口是 Cm 的
   matched policy utility，不是继续抬高无 Cm baseline。
2. 随机真实执行的 ±0.1 腕部动作在 seed151/152 上带来
   一步物体上抬 +12.19/+13.44mm，同时五步接触比例
   −7.58/−4.70pp。动作条件模型在未见 seed 能排序该权衡，
   但六区域几何未优于 raw 对照。
3. 两条固定在线单轴接法没有提高 held-lift：上抬 Cm
   86/128 vs base 87/128；接触感知下压在第一新 seed
   40/64 vs base 44/64，按预注册门提前停止。现有 PPO
   effect-rank 正式验证也未证明跨训练 seed 稳定增益。

Option A: 把冻结的短期物体/接触 Cm 用作 PPO 训练期
representation 或 auxiliary target，最终推理仍只用 actor。

成本：预计实现/工程门 2–4 小时；最小 matched 训练与评估
约 1 小时、1–2 GPU；若正向再做多训练 seed Validation。

成功后：做固定 Cm-on/off + 置乱/无效目标对照，争取正式
因果 policy-utility 证据。失败后：转 B 或重审 Cm 研究假设。

Option B: 扩展仿真随机干预到多个动作轴、更长随访，训练
新 Cm，再做 candidate control 或 planning。

成本：预计数据/模型/接法实现与 Probe 3–6 小时，1–2 GPU；
更大数据和规划延迟风险较高。

成功后：做 matched 在线 Cm-on/off Validation。失败后：
回 A 或重审动作模型路线。

User decision: A（2026-09-24）。在 `agent/cm-ppo-aux-representation`
实现训练期辅助表示 Probe；B 暂不启动。

AI recommendation: A。已有物理随机化数据表明局部效应
信息可学，但单轴贪心控制两次未变成完整抓取收益；训练期
辅助目标能让 actor 在较长时域内利用信息，且更快接近
MISSION 的 matched Cm-on/off 关键判定。A 仍是待检验假设，
不预设一定有效。
