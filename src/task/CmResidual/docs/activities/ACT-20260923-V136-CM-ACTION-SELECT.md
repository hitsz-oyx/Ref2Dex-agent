# V1.36 冻结 CmLite 的 s3 局部动作选择

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- evaluation code commit: `df01147`
- run_status: `COMPLETED`
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

Cm-off baseline `eval_s78_e180_full/run_manifest.json` 为
`COMPLETED`，严格抓取 32/64，转移 tensor 11,762,928 字节。
`cmlite_audit_v134.json` 正常产出；运动 EPE 相对零位移仅改善
18.6%，未过预注册的 20% 预检。按规则停止，未修改 DExplore
选择器代码，未做 selector 评估。GPU 5 已释放，无训练或删除。
