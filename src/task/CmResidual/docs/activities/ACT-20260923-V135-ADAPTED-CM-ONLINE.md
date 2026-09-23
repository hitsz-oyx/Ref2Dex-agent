# V1.35 适配 CmLite 的 s3 在线成对微调

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- training code commit: `81f0327`
- run_status: `RUNNING`
- official actor checkpoint: `null`

源为本地自训练 Cm-off s3 e160 actor，源 run manifest
`COMPLETED`，checkpoint SHA256
`ce62d6efb0c9a200583635eb942b45a07d972cfae098b8edb103e805a82cb8f1`。
新 CmLite 为 V1.34 平衡模型，SHA256
`1146025ca88b35f39a5fad7f5f899a88cca34b774fa9abe0c0d6f5d72519e1b8`。
两臂训练 seed70、s3 重建输入、64 env/horizon32/minibatch256；
Cm-on GPU5、Cm-off GPU6，各先 e162 smoke，再从同一源正式
训练至 e180。下一步评估 seed77 的 e170/e180 网格；详情与
停损规则见 experiment card。

工程 smoke 已启动：`agent_v135_s3_adaptcm_smoke_s70_e162`
占 GPU 5，`agent_v135_s3_cmoff_smoke_s70_e162` 占 GPU 6。
两臂使用相同源 e160 SHA、输入 manifest、seed70、64 env、
horizon32、minibatch256、学习率 `1e-5`，各只跑 2 epoch，
预算到 e162。各 run 的 `run_manifest.json` 记录精确命令、
输入/模型与资源；smoke 只验接线，不用于效果判定。

两个 smoke manifest 均为 `COMPLETED`；日志确认从同一 e160
checkpoint 恢复，`REF2DEX_LEARNING_RATE 1e-05`，Cm-on 产生
系数 1.0 的模型奖励。正式 run `agent_v135_s3_adaptcm_s70_e180`
（GPU 5）和 `agent_v135_s3_cmoff_s70_e180`（GPU 6）已各自
从**原始 e160** checkpoint 重新启动，epoch 预算到 180，
保存 e170/e180。新 run manifest 为 `STARTED`，工程 smoke
checkpoint 不参与正式训练。
