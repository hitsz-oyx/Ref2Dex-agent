# CmLite 在成功 scratch policy 上的局部动作选择

- experiment_id: `EXP-20260923-CMLITE-BC-PROPOSAL`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`

固定无官方 checkpoint 的 DAgger MLP 后，CmLite 适配模型在未见 seed 的真实转移上明显优于零位移基线，但在线五候选动作选择把最大抬升从 `0.1362 m` 降为 `0.1260 m`。这说明单步预测精度并不自动转化为闭环抓取增益，当前 selector 只能保留为后续反事实排序的工程入口。
