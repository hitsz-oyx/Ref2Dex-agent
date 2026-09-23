# V1.30 s1 自训练策略到 s3 的成对微调

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- code commit: `763014d`
- run_status: `RUNNING`
- official actor checkpoint: `null`

源为本地 scratch PPO epoch 140，SHA256
`5d1f50a21409df5a09f0a471d115d2f06eedce6bc53855a81acf3cc268e6e1a7`。
源 s1 训练 manifest 标记 `COMPLETED`，配置中没有恢复其他 actor；DExplore
scratch restore guard 进一步要求传入绝对路径与同一 SHA256。两臂使用
GPU 5/6、s3 重建输入 `corrected_manifest_r2.json`、seed 70、64 env、
horizon 32、学习率 `1e-5` 与相同课程参数。

- `agent_v130_s3_transfer_relcm_smoke_s70_e142`：GPU 5、源 epoch 140
  恢复到 142，`COMPLETED`；日志确认加载源 checkpoint 与学习率覆盖。
- `agent_v130_s3_transfer_cmoff_smoke_s70_e142`：GPU 6、相同恢复，
  `COMPLETED`。Cm-off 的 SHA 锁定恢复由 `f62807b` 加入，相关测试
  `14 passed`。
- `agent_v130_s3_transfer_relcm_s70_e160`：GPU 5，正式 20 epoch，
  run_status `STARTED`；CmLite 为 V1.29 相对手腕模型，SHA256
  `b7aa7630e31c820802cb95d81c490d4f27be8e74c1c9e1e400b20fcd849a9a38`。
- `agent_v130_s3_transfer_cmoff_s70_e160`：GPU 6，正式匹配对照，
  run_status `STARTED`。

正式运行的每个目录含 `run_manifest.json`、`config.json`、`train.log`；
源 e140 在 s3 seed 71 的完整轨迹基线为 2/64，产物在
`outputs/Dexplore/agent_v125_norm_cmlite_anneal_s45_e200/eval_s71_e140_full_s3_transfer`。
训练终态与 seed 71 的 e150/e160 结果待补，科学结论写对应 experiment card。

