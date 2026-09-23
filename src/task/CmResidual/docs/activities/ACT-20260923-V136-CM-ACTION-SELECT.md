# V1.36 冻结 CmLite 的 s3 局部动作选择

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- evaluation code commit: `df01147`
- run_status: `RUNNING`
- official actor checkpoint: `null`

冻结自训练 V1.35 Cm-off e180 PPO 与 V1.34 平衡 CmLite，
均锁定 SHA。不训练策略或模型。先在 seed78 完整轨迹评估
原策略并采集转移，再做离线 Cm 门禁；通过后才以同一 seed
评估五候选动作选择器。结果和预注册停损见 experiment card。

Cm-off baseline 已在 GPU 5 启动：run_id `eval_s78_e180_full`，
64 env、提前终止关闭、保存 `transitions.pt`。其
`run_manifest.json` 记录自训练 PPO checkpoint SHA、s3 输入
manifest SHA、精确命令和资源；完成后先审计 CmLite 的运动预测，
未通过不得运行选择器。
