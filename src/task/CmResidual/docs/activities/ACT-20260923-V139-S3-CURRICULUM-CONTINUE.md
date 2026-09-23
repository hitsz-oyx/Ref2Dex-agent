# V1.39 自训练 s3 策略继续训练与接触课程回启

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- training code commit: `791381d`
- run_status: `RUNNING`
- official actor checkpoint: `null`

共同源为 V1.35 自训练 Cm-off s3 e180，源 run manifest
`COMPLETED`、checkpoint SHA256
`863443513a746155f2b04662fe4dddd8c7b5a672d3e73533c1ff555d90c0237c`。
两臂均 Cm-off，不涉及官方 actor。GPU5 标准继续，GPU6
回启接触课程；先 e182 工程 smoke，之后从原 e180 各跑到
e260。评估/停损规则见 experiment card。

Cm-off 启动器原先未转发 `curriculum-backtrack`；已在
`791381d` 补齐参数配对/互斥校验、环境变量与配置记录，
`15 passed`，没有改变预注册课程。GPU5 的标准继续
`agent_v139_s3_standard_smoke_s70_e182` 与 GPU6 的课程回启
`agent_v139_s3_backtrack_smoke_s70_e182` 已从同一源 e180
SHA 启动两 epoch 接线 smoke，各自 manifest 初态 `STARTED`。
