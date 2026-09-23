# V1.33 用高抬升策略真实转移适配低延迟 CmLite

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- run_status: `COMPLETED`
- official actor checkpoint: `null`

V1.32 的 Cm-off e160 seed74 `eval_s74_e160_full/transitions.pt`
将筛为首 episode 非终止训练输入；seed75 的同一自训练策略评估
只用于独立门禁。训练还使用 V1.29 四个原始训练源，模型结构与
超参不变。完成后用 `analyze_cmlite_on_policy.py` 比较旧/新模型，
并用 `train_cmlite.py` 的 s1 外部验证检查遗忘。具体阈值见
experiment card。

已完成 seed74 Cm-off e160 筛选：33,189 条首 episode 非终止转移，
输出 `outputs/CmLite/V1.33/s3_cmoff_e160_seed74_first.pt`，SHA256
`e36c8116ae78772320de43150755d2f3cc6b170cbccef32f60ecc309611d066e`；
`s3_cmoff_e160_seed74_first.manifest.json` 为 `COMPLETED`。seed75
同一策略评估已在 GPU 5 启动，产物为 `eval_s75_e160_full`，不参与训练。

训练 run_id `s1_s3_highlift_seed74_train`，输出
`outputs/CmLite/V1.33/s1_s3_highlift_seed74_train`，GPU 6、
30 epochs、batch 2048、`3e-4`、seed 42。五个训练源 SHA256
依次为：s1 seed5910 `e18aacaf37c8b9b520d4feae080d2bfc12a99f2425a232a5bc8641bfedc8d7d2`，
s1 seed5929 `82325cdb958a991c9f66446c2191e82690456e49688fb638a0433758621c87c0`，
s3 e130 seed67 `2a2ee8e5a2dc1bf59d455201ba82834b826ed9d617d5bf87916f9f284d927a96`，
s3 e100 seed68 `db404622736fd2a45dc7b1f2dbdd3138aa2eba0cf1d0056bb5e8acb8e749c54c`，
本轮高抬升 seed74 为上述 `e36c...`。s1 外部验证 seed5909
SHA256 为 `a7790966180ab1765562bcb7af48cc8a113bd0f2f8ff6205d03ece2c4a41d65c`。
训练 checkpoint、逐 epoch `metrics.jsonl` 和最终 `summary.json`
都写入此 run 目录。

seed75 Cm-off e160 `eval_s75_e160_full/run_manifest.json` 为
`COMPLETED`：严格抓取 27/64，转移 tensor 11,762,928 字节，
只用于独立审计。GPU 6 上的训练正常结束 30 epoch；模型 SHA256
`2640f2e0b932be375f603bfecca4b3cb61310878b9a580e5512cecefa00272ee`，
最佳指标与 0.555 ms/64 样本前向见 `summary.json`，逐 epoch
指标在 `metrics.jsonl`，日志在 `train.log`。seed75 的旧/新
模型审计均完成，分别写为 `cmlite_audit_old.json` 与
`cmlite_audit_v133.json`；新模型 s3 运动预测过门，但 s1
seed5909 遗忘，故未上线。GPU 5/6 已释放。
